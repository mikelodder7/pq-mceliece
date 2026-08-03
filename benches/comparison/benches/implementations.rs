//! Cross-implementation Classic McEliece benchmarks.
//!
//! Each build selects exactly one parameter set because `classic-mceliece-rust` deliberately
//! rejects builds that enable more than one. See the adjacent README for run instructions.

use std::{hint::black_box, time::Duration};

use classic_mceliece_rust as classic;
use criterion::{BenchmarkId, Criterion};
use pq_mceliece::Algorithm;
use rand_chacha::ChaCha8Rng;
use rand_chacha_08::ChaCha8Rng as ChaCha8Rng08;

#[cfg(not(any(
    feature = "mceliece348864",
    feature = "mceliece348864f",
    feature = "mceliece460896",
    feature = "mceliece460896f",
    feature = "mceliece6688128",
    feature = "mceliece6688128f",
    feature = "mceliece6960119",
    feature = "mceliece6960119f",
    feature = "mceliece8192128",
    feature = "mceliece8192128f",
)))]
compile_error!("select one mceliece parameter-set feature");

#[cfg(any(
    all(
        feature = "mceliece348864",
        any(
            feature = "mceliece348864f",
            feature = "mceliece460896",
            feature = "mceliece460896f",
            feature = "mceliece6688128",
            feature = "mceliece6688128f",
            feature = "mceliece6960119",
            feature = "mceliece6960119f",
            feature = "mceliece8192128",
            feature = "mceliece8192128f",
        )
    ),
    all(
        feature = "mceliece348864f",
        any(
            feature = "mceliece460896",
            feature = "mceliece460896f",
            feature = "mceliece6688128",
            feature = "mceliece6688128f",
            feature = "mceliece6960119",
            feature = "mceliece6960119f",
            feature = "mceliece8192128",
            feature = "mceliece8192128f",
        )
    ),
    all(
        feature = "mceliece460896",
        any(
            feature = "mceliece460896f",
            feature = "mceliece6688128",
            feature = "mceliece6688128f",
            feature = "mceliece6960119",
            feature = "mceliece6960119f",
            feature = "mceliece8192128",
            feature = "mceliece8192128f",
        )
    ),
    all(
        feature = "mceliece460896f",
        any(
            feature = "mceliece6688128",
            feature = "mceliece6688128f",
            feature = "mceliece6960119",
            feature = "mceliece6960119f",
            feature = "mceliece8192128",
            feature = "mceliece8192128f",
        )
    ),
    all(
        feature = "mceliece6688128",
        any(
            feature = "mceliece6688128f",
            feature = "mceliece6960119",
            feature = "mceliece6960119f",
            feature = "mceliece8192128",
            feature = "mceliece8192128f",
        )
    ),
    all(
        feature = "mceliece6688128f",
        any(
            feature = "mceliece6960119",
            feature = "mceliece6960119f",
            feature = "mceliece8192128",
            feature = "mceliece8192128f",
        )
    ),
    all(
        feature = "mceliece6960119",
        any(
            feature = "mceliece6960119f",
            feature = "mceliece8192128",
            feature = "mceliece8192128f",
        )
    ),
    all(
        feature = "mceliece6960119f",
        any(feature = "mceliece8192128", feature = "mceliece8192128f")
    ),
    all(feature = "mceliece8192128", feature = "mceliece8192128f"),
))]
compile_error!("select exactly one mceliece parameter-set feature");

macro_rules! select_parameter_set {
    ($feature:literal, $variant:ident, $module:ident) => {
        #[cfg(feature = $feature)]
        const PARAMETER_SET: &str = $feature;
        #[cfg(feature = $feature)]
        const ALGORITHM: Algorithm = Algorithm::$variant;
        #[cfg(feature = $feature)]
        use pqcrypto_classicmceliece::$module as pqclean;
    };
}

select_parameter_set!("mceliece348864", McEliece348864, mceliece348864);
select_parameter_set!("mceliece348864f", McEliece348864f, mceliece348864f);
select_parameter_set!("mceliece460896", McEliece460896, mceliece460896);
select_parameter_set!("mceliece460896f", McEliece460896f, mceliece460896f);
select_parameter_set!("mceliece6688128", McEliece6688128, mceliece6688128);
select_parameter_set!("mceliece6688128f", McEliece6688128f, mceliece6688128f);
select_parameter_set!("mceliece6960119", McEliece6960119, mceliece6960119);
select_parameter_set!("mceliece6960119f", McEliece6960119f, mceliece6960119f);
select_parameter_set!("mceliece8192128", McEliece8192128, mceliece8192128);
select_parameter_set!("mceliece8192128f", McEliece8192128f, mceliece8192128f);

fn seeded_rng() -> ChaCha8Rng {
    <ChaCha8Rng as rand_core::SeedableRng>::from_seed([7; 32])
}

fn seeded_rng_08() -> ChaCha8Rng08 {
    <ChaCha8Rng08 as rand_08::SeedableRng>::from_seed([7; 32])
}

fn check_parameters() {
    let params = ALGORITHM.params();
    assert_eq!(params.name, PARAMETER_SET);
    assert_eq!(
        params.encapsulation_key_length,
        classic::CRYPTO_PUBLICKEYBYTES
    );
    assert_eq!(
        params.decapsulation_key_length,
        classic::CRYPTO_SECRETKEYBYTES
    );
    assert_eq!(params.ciphertext_length, classic::CRYPTO_CIPHERTEXTBYTES);
    assert_eq!(params.shared_secret_length, classic::CRYPTO_BYTES);
    assert_eq!(params.encapsulation_key_length, pqclean::public_key_bytes());
    assert_eq!(params.decapsulation_key_length, pqclean::secret_key_bytes());
    assert_eq!(params.ciphertext_length, pqclean::ciphertext_bytes());
    assert_eq!(params.shared_secret_length, pqclean::shared_secret_bytes());
}

fn keypair(c: &mut Criterion) {
    let mut group = c.benchmark_group("keypair");
    group.sample_size(10);

    let mut our_rng = seeded_rng();
    group.bench_function(BenchmarkId::new(PARAMETER_SET, "pq-mceliece"), |b| {
        b.iter(|| black_box(ALGORITHM.generate_keypair(&mut our_rng)));
    });

    let mut classic_rng = seeded_rng_08();
    group.bench_function(
        BenchmarkId::new(PARAMETER_SET, "classic-mceliece-rust"),
        |b| b.iter(|| black_box(classic::keypair_boxed(&mut classic_rng))),
    );

    group.bench_function(BenchmarkId::new(PARAMETER_SET, "PQClean"), |b| {
        b.iter(|| black_box(pqclean::keypair()));
    });
    group.finish();
}

fn encapsulate(c: &mut Criterion) {
    let mut our_rng = seeded_rng();
    let (our_pk, _) = ALGORITHM.generate_keypair(&mut our_rng);

    let mut classic_rng = seeded_rng_08();
    let (classic_pk, _) = classic::keypair_boxed(&mut classic_rng);

    let (pqclean_pk, _) = pqclean::keypair();

    let mut group = c.benchmark_group("encapsulate");
    group.bench_function(BenchmarkId::new(PARAMETER_SET, "pq-mceliece"), |b| {
        b.iter(|| {
            black_box(
                ALGORITHM
                    .encapsulate(&our_pk, &mut our_rng)
                    .expect("valid pq-mceliece public key"),
            )
        });
    });
    group.bench_function(
        BenchmarkId::new(PARAMETER_SET, "classic-mceliece-rust"),
        |b| b.iter(|| black_box(classic::encapsulate_boxed(&classic_pk, &mut classic_rng))),
    );
    group.bench_function(BenchmarkId::new(PARAMETER_SET, "PQClean"), |b| {
        b.iter(|| black_box(pqclean::encapsulate(&pqclean_pk)));
    });
    group.finish();
}

fn decapsulate(c: &mut Criterion) {
    let mut our_rng = seeded_rng();
    let (our_pk, our_sk) = ALGORITHM.generate_keypair(&mut our_rng);
    let (our_ct, our_sent) = ALGORITHM
        .encapsulate(&our_pk, &mut our_rng)
        .expect("valid pq-mceliece public key");
    assert_eq!(
        ALGORITHM
            .decapsulate(&our_sk, &our_ct)
            .expect("valid pq-mceliece ciphertext"),
        our_sent
    );

    let mut classic_rng = seeded_rng_08();
    let (classic_pk, classic_sk) = classic::keypair_boxed(&mut classic_rng);
    let (classic_ct, classic_sent) = classic::encapsulate_boxed(&classic_pk, &mut classic_rng);
    let classic_received = classic::decapsulate_boxed(&classic_ct, &classic_sk);
    assert_eq!(classic_received.as_array(), classic_sent.as_array());

    let (pqclean_pk, pqclean_sk) = pqclean::keypair();
    let (pqclean_sent, pqclean_ct) = pqclean::encapsulate(&pqclean_pk);
    assert_eq!(pqclean::decapsulate(&pqclean_ct, &pqclean_sk), pqclean_sent);

    let mut group = c.benchmark_group("decapsulate");
    group.bench_function(BenchmarkId::new(PARAMETER_SET, "pq-mceliece"), |b| {
        b.iter(|| {
            black_box(
                ALGORITHM
                    .decapsulate(&our_sk, &our_ct)
                    .expect("valid pq-mceliece ciphertext"),
            )
        });
    });
    group.bench_function(
        BenchmarkId::new(PARAMETER_SET, "classic-mceliece-rust"),
        |b| b.iter(|| black_box(classic::decapsulate_boxed(&classic_ct, &classic_sk))),
    );
    group.bench_function(BenchmarkId::new(PARAMETER_SET, "PQClean"), |b| {
        b.iter(|| black_box(pqclean::decapsulate(&pqclean_ct, &pqclean_sk)));
    });
    group.finish();
}

fn run() {
    check_parameters();

    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .configure_from_args();
    keypair(&mut criterion);
    encapsulate(&mut criterion);
    decapsulate(&mut criterion);
    criterion.final_summary();
}

fn main() {
    // PQClean's portable C implementation uses unusually large stack frames. Running all
    // implementations on this thread gives them identical scheduling conditions while avoiding
    // platform-dependent main-thread stack limits.
    std::thread::Builder::new()
        .name("mceliece-comparison".into())
        .stack_size(800_000_000)
        .spawn(run)
        .expect("create benchmark thread")
        .join()
        .expect("benchmark thread panicked");
}
