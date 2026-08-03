#!/bin/sh
set -eu

comparison_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

for parameter_set in \
    mceliece348864 mceliece348864f \
    mceliece460896 mceliece460896f \
    mceliece6688128 mceliece6688128f \
    mceliece6960119 mceliece6960119f \
    mceliece8192128 mceliece8192128f
do
    cargo bench \
        --manifest-path "$comparison_dir/Cargo.toml" \
        --no-default-features \
        --features "$parameter_set" \
        -- "$@"
done

