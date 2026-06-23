use std::fmt::Write as _;
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
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, imageops::FilterType};
use rayon::prelude::*;

const SIMPLE_CHARS: &[u8] = b" .:-=+*#%@";
const DETAILED_CHARS: &[u8] =
    b" .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
const CELL_ASPECT_RATIO: f32 = 0.5;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ColorMode {
    Truecolor,
    NoColor,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InputKind {
    Auto,
    Image,
    Video,
}

#[derive(Debug, Parser)]
#[command(
    name = "ascii-colorizer",
    about = "Convert images and videos to ASCII art"
)]
struct Cli {
    #[arg(short, long)]
    file: PathBuf,
    #[arg(short, long, default_value_t = 100)]
    width: u32,
    #[arg(long)]
    height: Option<u32>,
    #[arg(long)]
    detailed: bool,
    #[arg(long, value_enum, default_value_t = ColorMode::Truecolor)]
    color: ColorMode,
    #[arg(short, long)]
    save: Option<PathBuf>,
    #[arg(long = "type", value_enum, default_value_t = InputKind::Auto)]
    kind: InputKind,
    #[arg(long, default_value_t = 12.0)]
    fps: f32,
}

#[derive(Debug, Clone, Copy)]
struct AsciiConfig {
    charset: &'static [u8],
    width: u32,
    max_height: Option<u32>,
    color_mode: ColorMode,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
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

fn detect_input_kind(path: &Path) -> InputKind {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);

    match ext.as_deref() {
        Some("mp4" | "mkv" | "mov" | "avi" | "webm" | "flv" | "m4v") => InputKind::Video,
        _ => InputKind::Image,
    }
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

    loop {
        if interrupted.load(Ordering::SeqCst) {
            break;
        }
        let frame_started = Instant::now();

        match stdout.read_exact(&mut frame_buffer) {
            Ok(()) => {
                let rgb_image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                    config.width,
                    target_height,
                    frame_buffer,
                )
                .context("failed to decode ffmpeg raw frame")?;
                let frame = DynamicImage::ImageRgb8(rgb_image);
                // Video frames are already scaled by ffmpeg before shared rendering.
                render_resized_image_into(&frame, config, &mut ascii_buffer);
                frame_buffer = match frame {
                    DynamicImage::ImageRgb8(image) => image.into_raw(),
                    _ => unreachable!("frame is always ImageRgb8"),
                };

                if let Some(writer) = &mut file_writer {
                    if frame_index > 0 {
                        writer
                            .write_all(b"\n\x0C\n")
                            .context("failed to write frame separator")?;
                    }
                    writer
                        .write_all(ascii_buffer.as_bytes())
                        .context("failed to write frame output")?;
                } else {
                    if let Some(guard) = &mut terminal_guard {
                        guard.draw_frame(&ascii_buffer)?;
                    }

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
    let ffmpeg_status = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .status();
    let ffprobe_status = Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .status();

    if ffmpeg_status.is_err() || ffprobe_status.is_err() {
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
    let dims = stdout
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .trim_end_matches('x');
    let (w, h) = dims
        .split_once('x')
        .context("unexpected ffprobe output format for dimensions")?;

    let width = w.parse::<u32>().context("invalid width from ffprobe")?;
    let height = h.parse::<u32>().context("invalid height from ffprobe")?;
    Ok((width.max(1), height.max(1)))
}

fn render_ascii_image(image: &DynamicImage, config: AsciiConfig) -> String {
    let resized = resize_for_ascii(image, config.width, config.max_height);
    let mut output = String::new();
    render_resized_image_into(&resized, config, &mut output);
    output
}

fn flatten_rgba_over_black(r: u8, g: u8, b: u8, a: u8) -> (u8, u8, u8) {
    if a == u8::MAX {
        return (r, g, b);
    }

    let alpha = u16::from(a);
    let blend = |channel: u8| -> u8 { ((u16::from(channel) * alpha + 127) / 255) as u8 };
    (blend(r), blend(g), blend(b))
}

fn render_resized_image_into(image: &DynamicImage, config: AsciiConfig, output: &mut String) {
    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;
    let bytes_per_cell = if matches!(config.color_mode, ColorMode::Truecolor) {
        24
    } else {
        1
    };
    let estimated_len = height * (width * bytes_per_cell + 1);
    output.clear();
    if output.capacity() < estimated_len {
        output.reserve(estimated_len - output.capacity());
    }

    let rows: Vec<String> = (0..height as u32)
        .into_par_iter()
        .map(|y| {
            let mut row = String::with_capacity(width * bytes_per_cell);
            for x in 0..width as u32 {
                let [r, g, b, a] = image.get_pixel(x, y).0;
                let (r, g, b) = flatten_rgba_over_black(r, g, b, a);
                let symbol = map_luminance_to_char(r, g, b, config.charset);
                push_symbol(&mut row, symbol, (r, g, b), config.color_mode);
            }
            if matches!(config.color_mode, ColorMode::Truecolor) {
                row.push_str("\x1b[0m");
            }
            row.push('\n');
            row
        })
        .collect();

    for row in rows {
        output.push_str(&row);
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

fn compute_target_height(
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

fn map_luminance_to_char(r: u8, g: u8, b: u8, charset: &[u8]) -> char {
    let luminance = 0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b);
    let last = charset.len().saturating_sub(1);
    let idx = ((luminance / 255.0) * last as f32).round() as usize;

    char::from(*charset.get(idx).unwrap_or(&b' '))
}

fn push_symbol(buffer: &mut String, symbol: char, (r, g, b): (u8, u8, u8), color_mode: ColorMode) {
    match color_mode {
        ColorMode::Truecolor => {
            let _ = write!(buffer, "\x1b[38;2;{r};{g};{b}m{symbol}");
        }
        ColorMode::NoColor => buffer.push(symbol),
    }
}
