//! stories15M (Karpathy tinyllamas, Llama architecture) on the Rul fork
//! runtime - the head-to-head benchmark build against llama2.c's `run.c`.
//!
//! Usage: `cargo run --release --example stories15m [export_dir] [total_len]`
//! `export_dir` (default `stories15m_export`) must contain `weights.bin`,
//! `manifest.txt`, `step0_logits.bin`, `reference.json` from the lab
//! exporter (`~/workspace/rul-lab/bench/stories15m_export.py`).
//!
//! Contract (same split as the TinyStories runner): the host owns the
//! token loop and KV cache. Per token/layer it runs an RMSNorm Rul graph,
//! projects q/k/v on the host (RoPE is applied host-side to q and to the
//! new k before it enters the cache), then runs a block Rul graph with
//! attention over the cache, the output projection, a second RMSNorm and
//! the SwiGLU FFN (sigmoid composed from `Exp`). A final graph computes
//! the output RMSNorm + classifier. Greedy decoding, like `run.c -t 0`.

use rul::engine::{execute_tier1_stencil, RuntimeValue};
use rul::transpiler::assemble_rls;
use std::collections::HashMap;
use std::time::Instant;

const D: usize = 288;
const L: usize = 6;
const H: usize = 6;
const HD: usize = 48;
const EPS: f64 = 1e-5;

struct Weights {
    blob: Vec<f64>,
    idx: HashMap<String, (usize, usize)>,
}

impl Weights {
    fn t(&self, name: &str) -> &[f64] {
        let (off, len) = self.idx[name];
        &self.blob[off..off + len]
    }
}

fn read_f64s(path: &std::path::Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(bytes.len().is_multiple_of(8));
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn matvec(w: &[f64], x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0;
        for (a, b) in row.iter().zip(x.iter()) {
            acc += a * b;
        }
        out[r] = acc;
    }
    out
}

fn run_column(program: &rul::bytecode::RulProgram, inputs: &[&[f64]]) -> Vec<f64> {
    let rep = execute_tier1_stencil(program, inputs).expect("execute");
    match rep.output {
        RuntimeValue::ColumnF64(c) => c,
        other => panic!("expected column output, got {other:?}"),
    }
}

/// RMSNorm graph: arenas 0=x (d,), 1=gamma (d,) -> y (d,).
fn rms_rls() -> String {
    let mut s = String::from("cap 0x0\n");
    for a in 0..2 {
        s += &format!("arena {a} len={D} lane=f64x8 mut=false align=64 ver=1\n");
    }
    s += "node $x = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(0,1) a=0 b=0\n";
    s += "node $g = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(1,1) a=1 b=0\n";
    s += &format!("node $dc = ConstF64 f64x1 fg=1 pol=simd proof=0 tok=pure const={D}.0\n");
    s += &format!("node $eps = ConstF64 f64x1 fg=1 pol=simd proof=0 tok=pure const={EPS:e}\n");
    s += "node $sq = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$x b=$x\n";
    s += "node $ss = ReduceSum f64x1 fg=1 pol=simd proof=0 tok=pure a=$sq b=0\n";
    s += "node $ms = Div f64x1 fg=1 pol=simd proof=0 tok=pure a=$ss b=$dc\n";
    s += "node $me = Add f64x1 fg=1 pol=simd proof=0 tok=pure a=$ms b=$eps\n";
    s += "node $r = Rsqrt f64x1 fg=1 pol=simd proof=0 tok=pure a=$me b=0\n";
    s += "node $xn = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$x b=$r\n";
    s += "node $y = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$xn b=$g\n";
    s += "output $y\n";
    s
}

/// Block graph at context length t (cache already includes current k/v).
/// Inputs in order: x, q per head (HD each), K per head, V^T per head,
/// Wo per head, rms_ffn gamma, w1, w3, w2.
fn block_rls(t: usize) -> String {
    let mut s = String::from("cap 0x0\n");
    let mut arena = 0usize;
    let mut decl = |s: &mut String, len: usize| {
        *s += &format!("arena {arena} len={len} lane=f64x8 mut=false align=64 ver=1\n");
        arena += 1;
    };
    decl(&mut s, D);
    for _ in 0..H {
        decl(&mut s, HD);
    }
    for _ in 0..H {
        decl(&mut s, t * HD);
    }
    for _ in 0..H {
        decl(&mut s, HD * t);
    }
    for _ in 0..H {
        decl(&mut s, D * HD);
    }
    decl(&mut s, D);
    decl(&mut s, 768 * D);
    decl(&mut s, 768 * D);
    decl(&mut s, D * 768);
    let input = |s: &mut String, name: &str, a: usize| {
        *s += &format!(
            "node {name} = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read({a},1) a={a} b=0\n"
        );
    };
    input(&mut s, "$x", 0);
    let mut a = 1;
    for h in 0..H {
        input(&mut s, &format!("$q{h}"), a);
        a += 1;
    }
    for h in 0..H {
        input(&mut s, &format!("$k{h}"), a);
        a += 1;
    }
    for h in 0..H {
        input(&mut s, &format!("$vt{h}"), a);
        a += 1;
    }
    for h in 0..H {
        input(&mut s, &format!("$wo{h}"), a);
        a += 1;
    }
    for name in ["$rg", "$w1", "$w3", "$w2"] {
        input(&mut s, name, a);
        a += 1;
    }
    let konst = |s: &mut String, name: &str, v: f64| {
        *s += &format!("node {name} = ConstF64 f64x1 fg=1 pol=simd proof=0 tok=pure const={v}\n");
    };
    konst(&mut s, "$dc", 288.0);
    konst(&mut s, "$eps", EPS);
    konst(&mut s, "$scale", 6.928203230275509); // sqrt(48)
    konst(&mut s, "$one", 1.0);
    konst(&mut s, "$neg", -1.0);
    let node = |s: &mut String, name: &str, op: &str, aa: &str, b: &str| {
        *s += &format!("node {name} = {op} f64x8 fg=1 pol=simd proof=0 tok=pure a={aa} b={b}\n");
    };
    for h in 0..H {
        node(
            &mut s,
            &format!("$sc{h}"),
            "MatVec",
            &format!("$k{h}"),
            &format!("$q{h}"),
        );
        node(
            &mut s,
            &format!("$ss{h}"),
            "Div",
            &format!("$sc{h}"),
            "$scale",
        );
        node(
            &mut s,
            &format!("$mx{h}"),
            "ReduceMax",
            &format!("$ss{h}"),
            "0",
        );
        node(
            &mut s,
            &format!("$se{h}"),
            "Sub",
            &format!("$ss{h}"),
            &format!("$mx{h}"),
        );
        node(&mut s, &format!("$e{h}"), "Exp", &format!("$se{h}"), "0");
        node(
            &mut s,
            &format!("$es{h}"),
            "ReduceSum",
            &format!("$e{h}"),
            "0",
        );
        node(
            &mut s,
            &format!("$at{h}"),
            "Div",
            &format!("$e{h}"),
            &format!("$es{h}"),
        );
        node(
            &mut s,
            &format!("$o{h}"),
            "MatVec",
            &format!("$vt{h}"),
            &format!("$at{h}"),
        );
        node(
            &mut s,
            &format!("$p{h}"),
            "MatVec",
            &format!("$wo{h}"),
            &format!("$o{h}"),
        );
    }
    let mut acc = "$p0".to_string();
    for h in 1..H {
        let name = format!("$pa{h}");
        node(&mut s, &name, "Add", &acc, &format!("$p{h}"));
        acc = name;
    }
    node(&mut s, "$x1", "Add", "$x", &acc);
    // RMSNorm for FFN
    node(&mut s, "$sq", "Mul", "$x1", "$x1");
    node(&mut s, "$ssum", "ReduceSum", "$sq", "0");
    node(&mut s, "$ms", "Div", "$ssum", "$dc");
    node(&mut s, "$me", "Add", "$ms", "$eps");
    node(&mut s, "$r", "Rsqrt", "$me", "0");
    node(&mut s, "$xn", "Mul", "$x1", "$r");
    node(&mut s, "$z", "Mul", "$xn", "$rg");
    // SwiGLU: silu(w1 z) * (w3 z) -> w2
    node(&mut s, "$f1", "MatVec", "$w1", "$z");
    node(&mut s, "$f3", "MatVec", "$w3", "$z");
    node(&mut s, "$nf1", "Mul", "$f1", "$neg");
    node(&mut s, "$ef", "Exp", "$nf1", "0");
    node(&mut s, "$ed", "Add", "$ef", "$one");
    node(&mut s, "$sig", "Div", "$one", "$ed");
    node(&mut s, "$silu", "Mul", "$f1", "$sig");
    node(&mut s, "$h", "Mul", "$silu", "$f3");
    node(&mut s, "$m", "MatVec", "$w2", "$h");
    node(&mut s, "$out", "Add", "$x1", "$m");
    s += "output $out\n";
    s
}

/// Final graph: RMSNorm + classifier. Arenas 0=x, 1=gamma, 2=wcls.
fn final_rls() -> String {
    let mut s = rms_rls().replace("output $y\n", "");
    let wcls_decl = format!(
        "arena 2 len={} lane=f64x8 mut=false align=64 ver=1\n",
        32000 * D
    );
    s = s.replacen(
        "node $x = ArenaInput",
        &(wcls_decl + "node $x = ArenaInput"),
        1,
    );
    s += "node $wcls = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(2,1) a=2 b=0\n";
    s += "node $logits = MatVec f64x8 fg=1 pol=simd proof=0 tok=pure a=$wcls b=$y\n";
    s += "output $logits\n";
    s
}

fn extract_ids(json: &str, key: &str) -> Vec<u32> {
    let start = json.find(key).unwrap_or_else(|| panic!("{key} missing"));
    let open = json[start..].find('[').unwrap() + start;
    let close = json[open..].find(']').unwrap() + open;
    json[open + 1..close]
        .split(',')
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .collect()
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "stories15m_export".to_string());
    let total_len: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(260);
    let dir = std::path::Path::new(&dir);

    let blob = read_f64s(&dir.join("weights.bin"));
    let manifest = std::fs::read_to_string(dir.join("manifest.txt")).expect("manifest");
    let mut idx = HashMap::new();
    for line in manifest.lines() {
        let mut it = line.split_whitespace();
        let name = it.next().unwrap().to_string();
        let off: usize = it.next().unwrap().parse().unwrap();
        let len: usize = it.next().unwrap().parse().unwrap();
        idx.insert(name, (off, len));
    }
    let w = Weights { blob, idx };
    let ref_json = std::fs::read_to_string(dir.join("reference.json")).expect("reference.json");
    let prompt_ids = extract_ids(&ref_json, "\"prompt_ids\"");
    let ref_ids = extract_ids(&ref_json, "\"ids_to_260\"");
    let step0_ref = read_f64s(&dir.join("step0_logits.bin"));
    println!(
        "loaded {} tensors, prompt ids {:?}",
        w.idx.len(),
        prompt_ids
    );

    let rms_prog = assemble_rls(&rms_rls()).expect("assemble rms");
    let final_prog = assemble_rls(&final_rls()).expect("assemble final");
    let freq_re = w.t("freq_re").to_vec();
    let freq_im = w.t("freq_im").to_vec();

    let mut cache_k: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); H]; L];
    let mut cache_v: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); H]; L];

    let rope = |vec: &mut [f64], pos: usize, freq_re: &[f64], freq_im: &[f64]| {
        for hd in 0..H {
            let base = hd * HD;
            for i in 0..HD / 2 {
                let fcr = freq_re[pos * (HD / 2) + i];
                let fci = freq_im[pos * (HD / 2) + i];
                let a0 = vec[base + 2 * i];
                let a1 = vec[base + 2 * i + 1];
                vec[base + 2 * i] = a0 * fcr - a1 * fci;
                vec[base + 2 * i + 1] = a0 * fci + a1 * fcr;
            }
        }
    };

    let process_token = |pos: usize,
                         token: u32,
                         w: &Weights,
                         cache_k: &mut Vec<Vec<Vec<f64>>>,
                         cache_v: &mut Vec<Vec<Vec<f64>>>|
     -> Vec<f64> {
        let wte = w.t("wte");
        let mut x: Vec<f64> = wte[token as usize * D..(token as usize + 1) * D].to_vec();
        let t = pos + 1;
        for l in 0..L {
            let y = run_column(&rms_prog, &[x.as_slice(), w.t(&format!("l{l}_rms_att"))]);
            // host: full q/k/v projections with the layer's matrices
            // (rebuilt from per-head blocks in the manifest order)
            let mut q = vec![0.0f64; D];
            let mut knew = vec![0.0f64; D];
            let mut vnew = vec![0.0f64; D];
            for h in 0..H {
                q[h * HD..(h + 1) * HD].copy_from_slice(&matvec(
                    w.t(&format!("l{l}_wq{h}")),
                    &y,
                    HD,
                    D,
                ));
                knew[h * HD..(h + 1) * HD].copy_from_slice(&matvec(
                    w.t(&format!("l{l}_wk{h}")),
                    &y,
                    HD,
                    D,
                ));
                vnew[h * HD..(h + 1) * HD].copy_from_slice(&matvec(
                    w.t(&format!("l{l}_wv{h}")),
                    &y,
                    HD,
                    D,
                ));
            }
            rope(&mut q, pos, &freq_re, &freq_im);
            rope(&mut knew, pos, &freq_re, &freq_im);
            for h in 0..H {
                cache_k[l][h].extend_from_slice(&knew[h * HD..(h + 1) * HD]);
                cache_v[l][h].extend_from_slice(&vnew[h * HD..(h + 1) * HD]);
            }
            let prog = assemble_rls(&block_rls(t)).expect("assemble block");
            let mut inputs: Vec<&[f64]> = vec![x.as_slice()];
            for h in 0..H {
                inputs.push(&q[h * HD..(h + 1) * HD]);
            }
            for kh in &cache_k[l] {
                inputs.push(kh.as_slice());
            }
            let mut vt_bufs: Vec<Vec<f64>> = Vec::with_capacity(H);
            for v in &cache_v[l] {
                let mut vt = vec![0.0f64; HD * t];
                for i in 0..t {
                    for j in 0..HD {
                        vt[j * t + i] = v[i * HD + j];
                    }
                }
                vt_bufs.push(vt);
            }
            for vt in &vt_bufs {
                inputs.push(vt.as_slice());
            }
            for h in 0..H {
                inputs.push(w.t(&format!("l{l}_wo{h}")));
            }
            inputs.push(w.t(&format!("l{l}_rms_ffn")));
            inputs.push(w.t(&format!("l{l}_w1")));
            inputs.push(w.t(&format!("l{l}_w3")));
            inputs.push(w.t(&format!("l{l}_w2")));
            x = run_column(&prog, &inputs);
        }
        run_column(&final_prog, &[x.as_slice(), w.t("rms_final"), w.t("wcls")])
    };

    let argmax = |v: &[f64]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap()
    };

    let t_start = Instant::now();
    let mut ids: Vec<u32> = Vec::new();
    let mut logits: Vec<f64> = Vec::new();
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        ids.push(tok);
        logits = process_token(pos, tok, &w, &mut cache_k, &mut cache_v);
    }
    let max_diff = logits
        .iter()
        .zip(step0_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!(
        "step0 logits: max |rul - numpy| = {max_diff:.3e}; argmax {} vs {}",
        argmax(&logits),
        argmax(&step0_ref)
    );
    while ids.len() < total_len {
        let next = argmax(&logits);
        ids.push(next);
        if ids.len() >= total_len {
            break;
        }
        let pos = ids.len() - 1;
        logits = process_token(pos, next, &w, &mut cache_k, &mut cache_v);
    }
    let elapsed = t_start.elapsed();
    let matched = ids
        .iter()
        .zip(ref_ids.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "total {} tokens in {:?} -> {:.2} tok/s (llama2.c convention: (n-1)/t = {:.2})",
        ids.len(),
        elapsed,
        ids.len() as f64 / elapsed.as_secs_f64(),
        (ids.len() - 1) as f64 / elapsed.as_secs_f64()
    );
    println!(
        "greedy match vs numpy reference: first {matched} tokens identical (of {} compared)",
        ref_ids.len()
    );
    println!("GENERATED_IDS: {ids:?}");
}
