use anyhow::Result;
use ascii_colorizer::{Cli, run};
use clap::Parser;

fn main() -> Result<()> {
    run(Cli::parse())
}
