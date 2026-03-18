fn main() {
    // Re-run build if web/dist changes, so rust-embed picks up new assets
    println!("cargo:rerun-if-changed=web/dist");
}
