use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use rayon::prelude::*;

pub const SIMPLE_CHARS: &[u8] = b" .:-=+*#%@";
pub const DETAILED_CHARS: &[u8] =
    b" .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
pub const CELL_ASPECT_RATIO: f32 = 0.5;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum ColorMode {
    Truecolor,
    NoColor,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum InputKind {
    Auto,
    Image,
    Video,
}

#[derive(Debug, Parser)]
#[command(
    name = "ascii-colorizer",
    about = "Convert images and videos to ASCII art"
)]
pub struct Cli {
    #[arg(short, long)]
    pub file: PathBuf,
    #[arg(short, long, default_value_t = 100)]
    pub width: u32,
    #[arg(long)]
    pub height: Option<u32>,
    #[arg(long)]
    pub detailed: bool,
    #[arg(long, value_enum, default_value_t = ColorMode::Truecolor)]
    pub color: ColorMode,
    #[arg(short, long)]
    pub save: Option<PathBuf>,
    #[arg(long = "type", value_enum, default_value_t = InputKind::Auto)]
    pub kind: InputKind,
    #[arg(long, default_value_t = 12.0)]
    pub fps: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AsciiConfig {
    pub charset: &'static [u8],
    pub width: u32,
    pub max_height: Option<u32>,
    pub color_mode: ColorMode,
}

impl From<&Cli> for AsciiConfig {
    fn from(cli: &Cli) -> Self {
        Self {
            charset: if cli.detailed {
                DETAILED_CHARS
            } else {
                SIMPLE_CHARS
            },
            width: cli.width.max(1),
            max_height: cli.height,
            color_mode: cli.color,
        }
    }
}

struct RenderScratch {
    rows: Vec<Vec<u8>>,
}

impl RenderScratch {
    fn prepare(&mut self, height: u32, row_capacity: usize) {
        let height = height as usize;
        if self.rows.len() != height {
            self.rows = (0..height)
                .map(|_| Vec::with_capacity(row_capacity))
                .collect();
        }
        for row in &mut self.rows {
            row.clear();
            if row.capacity() < row_capacity {
                row.reserve(row_capacity - row.capacity());
            }
        }
    }
}

pub fn run(cli: Cli) -> Result<()> {
    let config = AsciiConfig::from(&cli);

    let kind = match cli.kind {
        InputKind::Auto => detect_input_kind(&cli.file),
        explicit => explicit,
    };

    match kind {
        InputKind::Image => {
            let image = image::open(&cli.file)
                .with_context(|| format!("failed to open image at {}", cli.file.display()))?;
            let rendered = render_ascii_image(&image, config);
            write_or_print(&rendered, cli.save.as_deref())?;
        }
        InputKind::Video => render_ascii_video(&cli.file, config, cli.fps, cli.save.as_deref())?,
        InputKind::Auto => unreachable!("auto kind is resolved above"),
    }

    Ok(())
}

fn write_or_print(content: &str, save_path: Option<&Path>) -> Result<()> {
    if let Some(output_path) = save_path {
        std::fs::write(output_path, content)
            .with_context(|| format!("failed to write output file {}", output_path.display()))?;
    } else {
        print!("{content}");
    }

    Ok(())
}

pub fn detect_input_kind(path: &Path) -> InputKind {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if is_video_extension(ext) => InputKind::Video,
        _ => InputKind::Image,
    }
}

fn is_video_extension(ext: &str) -> bool {
    ext.eq_ignore_ascii_case("mp4")
        || ext.eq_ignore_ascii_case("mkv")
        || ext.eq_ignore_ascii_case("mov")
        || ext.eq_ignore_ascii_case("avi")
        || ext.eq_ignore_ascii_case("webm")
        || ext.eq_ignore_ascii_case("flv")
        || ext.eq_ignore_ascii_case("m4v")
}

fn render_ascii_video(
    path: &Path,
    config: AsciiConfig,
    fps: f32,
    save_path: Option<&Path>,
) -> Result<()> {
    ensure_ffmpeg_tools_available()?;
    if fps <= 0.0 {
        bail!("fps must be greater than 0");
    }
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = Arc::clone(&interrupted);
        ctrlc::set_handler(move || {
            interrupted.store(true, Ordering::SeqCst);
        })
        .context("failed to install Ctrl+C handler")?;
    }

    let (src_width, src_height) = probe_video_dimensions(path)?;
    let target_height =
        compute_target_height(src_width, src_height, config.width, config.max_height);
    let frame_size = config.width as usize * target_height as usize * 3;

    let mut child = Command::new("ffmpeg")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(path)
        .arg("-vf")
        .arg(format!(
            "fps={fps},scale={}:{}",
            config.width, target_height
        ))
        .arg("-pix_fmt")
        .arg("rgb24")
        .arg("-f")
        .arg("rawvideo")
        .arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn ffmpeg process")?;

    let mut stdout = child
        .stdout
        .take()
        .context("failed to capture ffmpeg stdout")?;

    let mut file_writer = match save_path {
        Some(output_path) => Some(BufWriter::new(File::create(output_path).with_context(
            || format!("failed to create output file {}", output_path.display()),
        )?)),
        None => None,
    };

    let frame_duration = Duration::from_secs_f32(1.0 / fps);
    let mut terminal_guard = match save_path {
        Some(_) => None,
        None => Some(TerminalPlaybackGuard::enter()?),
    };
    let mut frame_index: usize = 0;
    let mut frame_buffer = vec![0_u8; frame_size];
    let mut ascii_buffer = String::new();
    let mut scratch = RenderScratch { rows: Vec::new() };

    loop {
        if interrupted.load(Ordering::SeqCst) {
            break;
        }
        let frame_started = Instant::now();

        match stdout.read_exact(&mut frame_buffer) {
            Ok(()) => {
                render_rgb24_into(
                    &frame_buffer,
                    config.width,
                    target_height,
                    config,
                    &mut scratch,
                    &mut ascii_buffer,
                );

                if let Some(writer) = &mut file_writer {
                    if frame_index > 0 {
                        writer
                            .write_all(b"\n\x0C\n")
                            .context("failed to write frame separator")?;
                    }
                    writer
                        .write_all(ascii_buffer.as_bytes())
                        .context("failed to write frame output")?;
                } else if let Some(guard) = &mut terminal_guard {
                    guard.draw_frame(&ascii_buffer)?;

                    let elapsed = frame_started.elapsed();
                    if elapsed < frame_duration {
                        std::thread::sleep(frame_duration - elapsed);
                    }
                }

                frame_index += 1;
            }
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => {
                return Err(error).context("error while reading frame stream from ffmpeg");
            }
        }
    }

    if let Some(writer) = &mut file_writer {
        writer.flush().context("failed to flush output file")?;
    }

    if let Some(mut guard) = terminal_guard.take() {
        guard.restore()?;
    }

    if interrupted.load(Ordering::SeqCst) {
        drop(stdout);
        let _ = child.kill();
    }

    let status = child.wait().context("failed to wait for ffmpeg process")?;
    if !status.success() && !interrupted.load(Ordering::SeqCst) {
        bail!("ffmpeg exited with status {status}");
    }

    Ok(())
}

struct TerminalPlaybackGuard {
    stdout: std::io::Stdout,
    restored: bool,
}

impl TerminalPlaybackGuard {
    fn enter() -> Result<Self> {
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, Hide)
            .context("failed to switch to alternate screen")?;
        Ok(Self {
            stdout,
            restored: false,
        })
    }

    fn draw_frame(&mut self, ascii: &str) -> Result<()> {
        execute!(self.stdout, MoveTo(0, 0)).context("failed to move cursor")?;
        self.stdout
            .write_all(ascii.as_bytes())
            .context("failed to write frame to terminal")?;
        self.stdout
            .flush()
            .context("failed to flush terminal output")
    }

    fn restore(&mut self) -> Result<()> {
        if !self.restored {
            execute!(self.stdout, Show, LeaveAlternateScreen)
                .context("failed to restore terminal state")?;
            self.restored = true;
        }
        Ok(())
    }
}

impl Drop for TerminalPlaybackGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

fn ensure_ffmpeg_tools_available() -> Result<()> {
    let ffmpeg_ok = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .status()
        .is_ok();
    let ffprobe_ok = Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .status()
        .is_ok();

    if !ffmpeg_ok || !ffprobe_ok {
        bail!("video mode requires ffmpeg and ffprobe on PATH");
    }

    Ok(())
}

fn probe_video_dimensions(path: &Path) -> Result<(u32, u32)> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=width,height")
        .arg("-of")
        .arg("csv=s=x:p=0")
        .arg(path)
        .output()
        .context("failed to run ffprobe")?;

    if !output.status.success() {
        bail!("ffprobe failed to inspect video stream");
    }

    let stdout = String::from_utf8(output.stdout).context("ffprobe emitted non utf-8 output")?;
    let dims = stdout.lines().next().unwrap_or("").trim();
    let (w, h) = dims
        .split_once('x')
        .context("unexpected ffprobe output format for dimensions")?;

    let width = w.parse::<u32>().context("invalid width from ffprobe")?;
    let height = h.parse::<u32>().context("invalid height from ffprobe")?;
    Ok((width.max(1), height.max(1)))
}

pub fn render_ascii_image(image: &DynamicImage, config: AsciiConfig) -> String {
    let resized = resize_for_ascii(image, config.width, config.max_height);
    let (width, height) = resized.dimensions();
    let rgba = resized.to_rgba8();
    let mut scratch = RenderScratch { rows: Vec::new() };
    let mut output = String::new();
    render_rgba8_into(
        rgba.as_raw(),
        width,
        height,
        config,
        &mut scratch,
        &mut output,
    );
    output
}

pub fn flatten_rgba_over_black(r: u8, g: u8, b: u8, a: u8) -> (u8, u8, u8) {
    if a == u8::MAX {
        return (r, g, b);
    }
    if a == 0 {
        return (0, 0, 0);
    }

    let alpha = u16::from(a);
    let blend = |channel: u8| -> u8 { ((u16::from(channel) * alpha + 127) / 255) as u8 };
    (blend(r), blend(g), blend(b))
}

pub fn render_resized_image_into(image: &DynamicImage, config: AsciiConfig, output: &mut String) {
    let (width, height) = image.dimensions();
    let rgba = image.to_rgba8();
    let mut scratch = RenderScratch { rows: Vec::new() };
    render_rgba8_into(rgba.as_raw(), width, height, config, &mut scratch, output);
}

fn render_rgb24_into(
    pixels: &[u8],
    width: u32,
    height: u32,
    config: AsciiConfig,
    scratch: &mut RenderScratch,
    output: &mut String,
) {
    let width = width as usize;
    let height = height as usize;
    let last = config.charset.len().saturating_sub(1);
    let row_capacity = row_byte_capacity(width, config.color_mode);
    scratch.prepare(height as u32, row_capacity);

    scratch
        .rows
        .par_iter_mut()
        .enumerate()
        .for_each(|(y, row)| {
            let row_base = y * width * 3;
            for x in 0..width {
                let i = row_base + x * 3;
                let (r, g, b) = (pixels[i], pixels[i + 1], pixels[i + 2]);
                let ch = charset_char_at_luminance(r, g, b, config.charset, last);
                append_cell(row, ch, (r, g, b), config.color_mode);
            }
            finish_row(row, config.color_mode);
        });

    assemble_rows(scratch, output);
}

fn render_rgba8_into(
    pixels: &[u8],
    width: u32,
    height: u32,
    config: AsciiConfig,
    scratch: &mut RenderScratch,
    output: &mut String,
) {
    let width = width as usize;
    let height = height as usize;
    let last = config.charset.len().saturating_sub(1);
    let row_capacity = row_byte_capacity(width, config.color_mode);
    scratch.prepare(height as u32, row_capacity);

    scratch
        .rows
        .par_iter_mut()
        .enumerate()
        .for_each(|(y, row)| {
            let row_base = y * width * 4;
            for x in 0..width {
                let i = row_base + x * 4;
                let (r, g, b) =
                    flatten_rgba_over_black(pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]);
                let ch = charset_char_at_luminance(r, g, b, config.charset, last);
                append_cell(row, ch, (r, g, b), config.color_mode);
            }
            finish_row(row, config.color_mode);
        });

    assemble_rows(scratch, output);
}

fn assemble_rows(scratch: &RenderScratch, output: &mut String) {
    let total: usize = scratch.rows.iter().map(Vec::len).sum();
    output.clear();
    output.reserve(total);
    for row in &scratch.rows {
        // SAFETY: rows contain only ASCII escapes and single-byte UTF-8 chars.
        unsafe {
            output.push_str(std::str::from_utf8_unchecked(row));
        }
    }
}

fn row_byte_capacity(width: usize, color_mode: ColorMode) -> usize {
    match color_mode {
        ColorMode::Truecolor => width * 20 + 5,
        ColorMode::NoColor => width + 1,
    }
}

fn finish_row(row: &mut Vec<u8>, color_mode: ColorMode) {
    if matches!(color_mode, ColorMode::Truecolor) {
        row.extend_from_slice(b"\x1b[0m\n");
    } else {
        row.push(b'\n');
    }
}

#[inline]
fn charset_char_at_luminance(r: u8, g: u8, b: u8, charset: &[u8], last: usize) -> u8 {
    let idx = luminance_index(r, g, b, last);
    *charset.get(idx).unwrap_or(&b' ')
}

#[inline]
fn luminance_index(r: u8, g: u8, b: u8, last: usize) -> usize {
    let luminance = 0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b);
    ((luminance / 255.0) * last as f32).round() as usize
}

fn append_cell(row: &mut Vec<u8>, ch: u8, (r, g, b): (u8, u8, u8), color_mode: ColorMode) {
    match color_mode {
        ColorMode::Truecolor => append_truecolor(row, r, g, b, ch),
        ColorMode::NoColor => row.push(ch),
    }
}

fn append_truecolor(row: &mut Vec<u8>, r: u8, g: u8, b: u8, ch: u8) {
    row.extend_from_slice(b"\x1b[38;2;");
    append_decimal(row, r);
    row.push(b';');
    append_decimal(row, g);
    row.push(b';');
    append_decimal(row, b);
    row.extend_from_slice(b"m");
    row.push(ch);
}

fn append_decimal(row: &mut Vec<u8>, mut n: u8) {
    if n >= 100 {
        row.push(b'0' + n / 100);
        n %= 100;
        row.push(b'0' + n / 10);
        row.push(b'0' + n % 10);
    } else if n >= 10 {
        row.push(b'0' + n / 10);
        row.push(b'0' + n % 10);
    } else {
        row.push(b'0' + n);
    }
}

fn resize_for_ascii(
    image: &DynamicImage,
    target_width: u32,
    max_height: Option<u32>,
) -> DynamicImage {
    let (src_w, src_h) = image.dimensions();
    let target_height = compute_target_height(src_w, src_h, target_width, max_height);
    image.resize_exact(target_width.max(1), target_height, FilterType::Triangle)
}

pub fn compute_target_height(
    src_w: u32,
    src_h: u32,
    target_width: u32,
    max_height: Option<u32>,
) -> u32 {
    let src_w = src_w.max(1);
    let aspect_ratio = src_h as f32 / src_w as f32;
    let computed_height = ((target_width.max(1) as f32) * aspect_ratio * CELL_ASPECT_RATIO)
        .round()
        .max(1.0) as u32;

    max_height
        .map(|limit| computed_height.min(limit.max(1)))
        .unwrap_or(computed_height)
}

pub fn map_luminance_to_char(r: u8, g: u8, b: u8, charset: &[u8]) -> char {
    let last = charset.len().saturating_sub(1);
    char::from(charset_char_at_luminance(r, g, b, charset, last))
}

pub fn push_symbol(
    buffer: &mut String,
    symbol: char,
    (r, g, b): (u8, u8, u8),
    color_mode: ColorMode,
) {
    let mut row = Vec::new();
    append_cell(&mut row, symbol as u8, (r, g, b), color_mode);
    buffer.push_str(unsafe { std::str::from_utf8_unchecked(&row) });
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    fn solid_rgba(w: u32, h: u32, pixel: Rgba<u8>) -> DynamicImage {
        DynamicImage::ImageRgba8(ImageBuffer::from_pixel(w, h, pixel))
    }

    fn default_config() -> AsciiConfig {
        AsciiConfig {
            charset: SIMPLE_CHARS,
            width: 4,
            max_height: None,
            color_mode: ColorMode::NoColor,
        }
    }

    #[test]
    fn detect_input_kind_video_extensions() {
        for ext in [
            "mp4", "mkv", "mov", "avi", "webm", "flv", "m4v", "MP4", "WebM",
        ] {
            let path = Path::new("clip").with_extension(ext);
            assert_eq!(detect_input_kind(&path), InputKind::Video, "ext: {ext}");
        }
    }

    #[test]
    fn detect_input_kind_image_and_unknown() {
        assert_eq!(detect_input_kind(Path::new("photo.png")), InputKind::Image);
        assert_eq!(detect_input_kind(Path::new("photo.jpeg")), InputKind::Image);
        assert_eq!(detect_input_kind(Path::new("noext")), InputKind::Image);
    }

    #[test]
    fn compute_target_height_preserves_terminal_aspect_ratio() {
        assert_eq!(compute_target_height(1920, 1080, 100, None), 28);
    }

    #[test]
    fn compute_target_height_clamps_zero_dimensions() {
        assert_eq!(compute_target_height(0, 0, 10, None), 1);
        assert_eq!(compute_target_height(0, 500, 10, None), 2500);
    }

    #[test]
    fn compute_target_height_respects_max_height() {
        assert_eq!(compute_target_height(100, 100, 100, Some(5)), 5);
        assert_eq!(compute_target_height(100, 100, 100, Some(0)), 1);
    }

    #[test]
    fn compute_target_height_width_zero_uses_one() {
        assert_eq!(compute_target_height(200, 100, 0, None), 1);
    }

    #[test]
    fn flatten_rgba_opaque_passthrough() {
        assert_eq!(flatten_rgba_over_black(10, 20, 30, 255), (10, 20, 30));
    }

    #[test]
    fn flatten_rgba_fully_transparent_is_black() {
        assert_eq!(flatten_rgba_over_black(255, 128, 64, 0), (0, 0, 0));
    }

    #[test]
    fn flatten_rgba_half_alpha_blends_toward_black() {
        let (r, g, b) = flatten_rgba_over_black(200, 100, 50, 128);
        assert_eq!(r, 100);
        assert_eq!(g, 50);
        assert_eq!(b, 25);
    }

    fn reference_luminance_index(r: u8, g: u8, b: u8, last: usize) -> usize {
        let luminance = 0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b);
        ((luminance / 255.0) * last as f32).round() as usize
    }

    #[test]
    fn luminance_index_regression_triples() {
        let last = SIMPLE_CHARS.len() - 1;
        let cases = [
            ((0, 0, 0), 0),
            ((255, 255, 255), last),
            ((0, 4, 157), 1),
            ((0, 74, 248), 2),
            ((255, 0, 0), 2),
            ((0, 255, 0), 6),
        ];
        for ((r, g, b), expected) in cases {
            assert_eq!(
                luminance_index(r, g, b, last),
                expected,
                "rgb=({r},{g},{b})"
            );
        }
    }

    #[test]
    fn luminance_index_matches_bt709_reference() {
        let last = SIMPLE_CHARS.len() - 1;
        for r in 0..=255_u8 {
            for g in 0..=255_u8 {
                for b in 0..=255_u8 {
                    if (u16::from(r) + u16::from(g) + u16::from(b)) % 17 != 0 {
                        continue;
                    }
                    assert_eq!(
                        luminance_index(r, g, b, last),
                        reference_luminance_index(r, g, b, last),
                        "rgb=({r},{g},{b})"
                    );
                }
            }
        }
    }

    #[test]
    fn map_luminance_black_and_white() {
        assert_eq!(
            map_luminance_to_char(0, 0, 0, SIMPLE_CHARS),
            char::from(SIMPLE_CHARS[0])
        );
        let last = SIMPLE_CHARS.len() - 1;
        assert_eq!(
            map_luminance_to_char(255, 255, 255, SIMPLE_CHARS),
            char::from(SIMPLE_CHARS[last])
        );
    }

    #[test]
    fn map_luminance_green_weighted() {
        let dark = map_luminance_to_char(0, 0, 0, SIMPLE_CHARS);
        let bright_green = map_luminance_to_char(0, 255, 0, SIMPLE_CHARS);
        assert_ne!(dark, bright_green);
    }

    #[test]
    fn push_symbol_truecolor_and_plain() {
        let mut colored = String::new();
        push_symbol(&mut colored, '@', (1, 2, 3), ColorMode::Truecolor);
        assert_eq!(colored, "\x1b[38;2;1;2;3m@");

        let mut plain = String::new();
        push_symbol(&mut plain, '#', (9, 9, 9), ColorMode::NoColor);
        assert_eq!(plain, "#");
    }

    #[test]
    fn render_solid_black_no_color() {
        let image = solid_rgba(2, 2, Rgba([0, 0, 0, 255]));
        let config = AsciiConfig {
            width: 2,
            max_height: Some(2),
            ..default_config()
        };
        let output = render_ascii_image(&image, config);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "  ");
    }

    #[test]
    fn render_solid_white_no_color() {
        let image = solid_rgba(1, 1, Rgba([255, 255, 255, 255]));
        let config = AsciiConfig {
            width: 1,
            max_height: Some(1),
            ..default_config()
        };
        let output = render_ascii_image(&image, config);
        let expected = char::from(*SIMPLE_CHARS.last().unwrap());
        assert_eq!(output.trim_end_matches('\n'), expected.to_string());
    }

    #[test]
    fn render_truecolor_resets_each_row() {
        let image = solid_rgba(2, 1, Rgba([255, 0, 0, 255]));
        let config = AsciiConfig {
            width: 2,
            max_height: Some(1),
            color_mode: ColorMode::Truecolor,
            ..default_config()
        };
        let output = render_ascii_image(&image, config);
        assert!(output.contains("\x1b[38;2;"));
        assert!(output.ends_with("\x1b[0m\n"));
    }

    #[test]
    fn render_transparent_pixel_counts_as_black() {
        let image = solid_rgba(1, 1, Rgba([255, 255, 255, 0]));
        let config = AsciiConfig {
            width: 1,
            max_height: Some(1),
            ..default_config()
        };
        let output = render_ascii_image(&image, config);
        assert_eq!(output.trim_end_matches('\n'), " ");
    }

    #[test]
    fn render_rgb24_solid_red() {
        let pixels = [255_u8, 0, 0];
        let mut scratch = RenderScratch { rows: Vec::new() };
        let mut output = String::new();
        render_rgb24_into(
            &pixels,
            1,
            1,
            AsciiConfig {
                width: 1,
                max_height: Some(1),
                color_mode: ColorMode::NoColor,
                ..default_config()
            },
            &mut scratch,
            &mut output,
        );
        assert_eq!(output.trim_end_matches('\n').len(), 1);
    }

    #[test]
    fn render_resized_image_into_reuses_buffer() {
        let image = solid_rgba(1, 1, Rgba([128, 128, 128, 255]));
        let config = AsciiConfig {
            width: 1,
            max_height: Some(1),
            ..default_config()
        };
        let mut buf = String::from("previous content");
        render_resized_image_into(&image, config, &mut buf);
        assert!(!buf.contains("previous"));
        assert_eq!(buf.lines().count(), 1);
    }

    #[test]
    fn render_scratch_reuses_row_allocations() {
        let pixels = vec![128_u8; 3 * 2 * 2];
        let config = AsciiConfig {
            width: 2,
            max_height: Some(2),
            ..default_config()
        };
        let mut scratch = RenderScratch { rows: Vec::new() };
        let mut output = String::new();
        render_rgb24_into(&pixels, 2, 2, config, &mut scratch, &mut output);
        let caps: Vec<_> = scratch.rows.iter().map(|r| r.capacity()).collect();
        render_rgb24_into(&pixels, 2, 2, config, &mut scratch, &mut output);
        assert_eq!(
            scratch
                .rows
                .iter()
                .map(|r| r.capacity())
                .collect::<Vec<_>>(),
            caps
        );
    }

    #[test]
    fn render_rgb24_two_column_row() {
        let pixels = [255_u8, 0, 0, 0, 255, 0];
        let mut scratch = RenderScratch { rows: Vec::new() };
        let mut output = String::new();
        render_rgb24_into(
            &pixels,
            2,
            1,
            AsciiConfig {
                width: 2,
                max_height: Some(1),
                ..default_config()
            },
            &mut scratch,
            &mut output,
        );
        assert_eq!(output.lines().next().unwrap().chars().count(), 2);
    }

    #[test]
    fn ascii_config_from_cli_clamps_width() {
        let cli = Cli {
            file: PathBuf::from("x.png"),
            width: 0,
            height: None,
            detailed: false,
            color: ColorMode::NoColor,
            save: None,
            kind: InputKind::Image,
            fps: 12.0,
        };
        assert_eq!(AsciiConfig::from(&cli).width, 1);
    }

    #[test]
    fn charsets_are_non_empty_and_monotonic_luminance() {
        assert!(!SIMPLE_CHARS.is_empty());
        assert!(!DETAILED_CHARS.is_empty());
        assert!(DETAILED_CHARS.len() > SIMPLE_CHARS.len());
    }
}
