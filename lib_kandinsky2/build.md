# Requirements to build

```
git+https://github.com/ai-forever/Kandinsky-2.git
transformers==4.23.1
huggingface-hub==0.14.0
```


## tokenizers==0.13.3 build with rust

```bash
sudo apt install rustc cargo rustup
```

### Fix base64ct build
The error indicates that the `base64ct` crate requires the `edition2024` feature, which is not supported in your current version of Cargo (`1.75.0`). To fix this, you need to update Cargo to a nightly version that supports the `edition2024` feature.

Here’s how to fix it:

#### 1. Install Rust Nightly
Run the following command to install the nightly version of Rust:
```bash
rustup install nightly
```

#### 2. Use Nightly for Your Project
Switch to the nightly version of Rust for your current project:
```bash
rustup override set nightly
```

Alternatively, you can use nightly globally:
```bash
rustup default nightly
```

#### 3. Verify the Cargo Version
Check that you are now using a nightly version of Cargo:
```bash
cargo --version
```
You should see something like `cargo 1.x.x-nightly`.

#### 4. Retry the Command
Run the command that failed again:
```bash
cargo metadata --manifest-path Cargo.toml --format-version 1
```

This should resolve the issue. If you encounter further problems, ensure all dependencies are compatible with the nightly version.