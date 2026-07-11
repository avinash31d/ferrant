fn main() {
    if let Err(error) = ferrant::cli::run_from_env() {
        eprintln!("ferrant: {error:#}");
        std::process::exit(1);
    }
}
