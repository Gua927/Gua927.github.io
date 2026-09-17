// Co-authorship network, seeded from SAT-Mask and grown one hop outward.
//
// Edges are not written by hand — they are derived from `papers`, so two people
// are linked exactly when they share a paper listed below. Adding a paper is
// one entry and the graph updates itself.
//
// The layout is a force simulation that runs ONCE, here, at build time. The
// coordinates are baked into the HTML, so every visitor sees the identical
// picture and the browser ships no physics at all. Only the camera moves.

import { existsSync } from "node:fs";
import { join } from "node:path";

export type GroupKey =
  | "hub"
  | "air"
  | "dlm"
  | "sbdd"
  | "protein"
  | "cityu"
  | "pku"
  | "stanford"
  | "mila"
  | "westlake";

export type Person = {
  id: string;
  name: string;
  affiliation: string;
  /** Shown in the caption when this person is selected. Two sentences, tops. */
  bio: string;
  group: GroupKey;
  href?: string;
  /** Local path under /public. Falls back to initials when absent. */
  avatar?: string;
};

export type Paper = {
  id: string;
  title: string;
  venue: string;
  /** Person ids. Authors outside this roster are simply omitted. */
  authors: string[];
};

export const me: Person = {
  id: "me",
  name: "Runze Tian",
  affiliation: "RUC Statistics · GenSI Lab, THU-AIR",
  bio:
    "Undergraduate in Statistics at Renmin University of China. Research intern with GenSI at " +
    "THU-AIR, and with Tailin Wu's group at Westlake University from 2026. Works on generative " +
    "models — diffusion language models, and how a generation trajectory meets the structure of " +
    "the data it is drawing from.",
  group: "hub",
  avatar: "/assets/img/prof_pic.jpg",
};

export const people: Person[] = [
  // ── THU-AIR, faculty ───────────────────────────────────────────────
  {
    id: "hao-zhou",
    name: "Hao Zhou",
    affiliation: "Research Assoc. Prof, THU-AIR · leads GenSI",
    bio:
      "Research Associate Professor at THU-AIR, where he leads the GenSI group; before that a " +
      "researcher at ByteDance AI Lab. Works on generative modelling for language and for " +
      "biomolecules, and is senior author on seven of the papers behind this map.",
    group: "air",
    href: "https://zhouh.github.io/",
  },
  {
    id: "weiying-ma",
    name: "Wei-Ying Ma",
    affiliation: "Chief Scientist, THU-AIR · Chief of AI, CityU",
    bio:
      "Huiyan Chair Professor and Chief Scientist at THU-AIR; now also Chair Professor of AI for " +
      "Science in CityU's Department of Computer Science, the university's Chief of AI, and " +
      "Director of the Hong Kong Institute of AI for Science, where Ning Miao's group sits. A " +
      "long-standing figure in information retrieval and multimedia who co-advises much of AIR's " +
      "generative work; co-author on six papers here.",
    group: "air",
    href: "https://air.tsinghua.edu.cn/en/info/1046/1189.htm",
  },
  {
    id: "yaqin-zhang",
    name: "Ya-Qin Zhang",
    affiliation: "Founding Dean & Chair Professor, THU-AIR",
    bio:
      "Founding Dean and Chair Professor of THU-AIR, and Dean of Tsinghua's Institute for AI " +
      "Industry Research. Previously President of Baidu and a Corporate Vice President at " +
      "Microsoft.",
    group: "air",
    href: "https://air.tsinghua.edu.cn/en/info/1046/1188.htm",
  },

  // ── diffusion language models ───────────────────────────────────────────
  {
    id: "yuxuan-song",
    name: "Yuxuan Song",
    affiliation: "Research Scientist, ByteDance Seed · PhD, THU-AIR",
    bio:
      "Research Scientist at ByteDance Seed and a THU-AIR PhD. Works on diffusion and flow-based " +
      "generative models across language and molecules, including Seed Diffusion; co-author on " +
      "nine papers here, more than anyone else on the map.",
    group: "dlm",
    href: "https://yuxuansong.com/",
  },
  {
    id: "zhilong-zhang",
    name: "Zhilong Zhang",
    affiliation: "PhD student, THU-AIR",
    bio:
      "PhD student at THU-AIR, working on diffusion language models and on structure-based " +
      "molecule design; co-author on SAT-Mask. Before that a master's student at Peking " +
      "University, where he co-wrote the diffusion-models survey and a string of diffusion papers " +
      "with Ling Yang in Bin Cui's lab.",
    group: "dlm",
    href: "https://scholar.google.com/citations?user=irQZ_qgAAAAJ",
  },

  // ── structure-based drug design / molecules ─────────────────────────────
  {
    id: "keyue-qiu",
    name: "Keyue Qiu",
    affiliation: "PhD student, THU-AIR",
    bio:
      "PhD student at THU-AIR, on seven of the papers listed here. Works on structure-based " +
      "drug design and multimodal biomolecular co-design.",
    group: "sbdd",
    href: "https://qky18.github.io/",
  },

  // ── CityU, by way of ByteDance AI Lab ───────────────────────────────────
  {
    id: "ning-miao",
    name: "Ning Miao",
    affiliation: "Assistant Professor, Data Science & HKAI-Sci, CityU",
    bio:
      "Assistant Professor at City University of Hong Kong, in the Department of Data Science and " +
      "the Hong Kong Institute of AI for Science. Was Hao Zhou's colleague at ByteDance AI Lab, " +
      "working on constrained and controllable text generation with him and Yuxuan Song, then did " +
      "a PhD at Oxford on getting language models to check their own reasoning.",
    group: "cityu",
    href: "https://www.ningmiao.space/",
  },

  // ── Peking University, where Zhilong Zhang did his master's ─────────────
  // PKU-DAIR is Bin Cui's lab; Ling Yang led the diffusion work there that
  // Zhilong Zhang grew up on. Yang Song and Stefano Ermon come in through the
  // same papers and through Ermon having advised Song's PhD.
  {
    id: "ling-yang",
    name: "Ling Yang",
    affiliation: "Incoming Asst. Prof, PKU · Postdoc, Princeton",
    bio:
      "Incoming Assistant Professor at Peking University, currently a postdoc at Princeton with " +
      "Mengdi Wang. Did his PhD at PKU with Bin Cui; first author of the diffusion-models survey " +
      "and of most of Zhilong Zhang's PKU-era papers, now working on LLM post-training and " +
      "recursive self-improvement.",
    group: "pku",
    href: "https://yangling0818.github.io/",
  },
  {
    id: "yang-song",
    name: "Yang Song",
    affiliation: "Research Principal, Meta Superintelligence Labs",
    bio:
      "Research Principal at Meta Superintelligence Labs, previously at OpenAI. His Stanford PhD " +
      "with Stefano Ermon produced score-based diffusion models — the foundation of much of what " +
      "this map works on — and he co-authored the diffusion-models survey.",
    group: "pku",
    href: "https://yang-song.net/",
  },
  {
    id: "bin-cui",
    name: "Bin Cui",
    affiliation: "Boya Distinguished Professor & Vice Dean, School of CS, PKU",
    bio:
      "Boya Distinguished Professor and Vice Dean of the School of Computer Science at Peking " +
      "University, leading the PKU-DAIR lab. Works on database and AI systems; senior author on " +
      "the diffusion papers Zhilong Zhang wrote there as a master's student.",
    group: "pku",
    href: "https://cuibinpku.github.io/",
  },
  {
    id: "wentao-zhang",
    name: "Wentao Zhang",
    affiliation: "Assistant Professor, PKU · Data-Centric AI",
    bio:
      "Assistant Professor at Peking University, leading the Data-Centric AI group; PhD at PKU " +
      "with Bin Cui, then Mila, Tencent and Apple. Works on data-centric AI and LLM data systems.",
    group: "pku",
    href: "https://zwt233.github.io/",
  },
  {
    id: "stefano-ermon",
    name: "Stefano Ermon",
    affiliation: "Associate Professor of CS, Stanford",
    bio:
      "Associate Professor of Computer Science at Stanford. Works on machine learning and " +
      "generative AI; advised Yang Song's PhD and co-advised Minkai Xu's, and co-authored two of " +
      "the PKU diffusion papers here.",
    group: "pku",
    href: "https://cs.stanford.edu/~ermon/",
  },

  // ── Stanford and Mila, by way of Minkai Xu ──────────────────────────────
  // Minkai Xu co-wrote two of the PKU papers with Zhilong Zhang, and his own
  // advisers pull in the other direction: Ermon and Leskovec at Stanford, Jian
  // Tang at Mila before that. Leskovec is already on the map through Tailin Wu,
  // so this is where the Westlake side and the PKU side meet.
  {
    id: "minkai-xu",
    name: "Minkai Xu",
    affiliation: "Research Scientist, Google DeepMind · PhD, Stanford",
    bio:
      "Research Scientist at Google DeepMind. His Stanford PhD with Stefano Ermon and Jure " +
      "Leskovec, after a master's at Mila with Jian Tang, produced GeoDiff and GeoLDM — diffusion " +
      "models for molecular geometry — and he co-wrote ContextDiff and Consistency Flow Matching " +
      "with Zhilong Zhang.",
    group: "stanford",
    href: "https://minkaixu.com/",
  },
  {
    id: "jian-tang",
    name: "Jian Tang",
    affiliation: "Associate Professor, Mila & HEC Montréal",
    bio:
      "Associate Professor at Mila and HEC Montréal. Works on graph neural networks and " +
      "generative models for drug discovery and molecular design; advised Minkai Xu's master's, " +
      "and is senior author on GeoDiff.",
    group: "mila",
    href: "https://jian-tang.com/",
  },

  // ── Westlake, and the lineage behind it ─────────────────────────────────
  // The group key is a layout device, not a claim of affiliation: it keeps
  // this branch together on the canvas. The affiliation lines carry the truth.
  {
    id: "tailin-wu",
    name: "Tailin Wu",
    affiliation: "Assistant Professor, Westlake University",
    bio:
      "Assistant Professor at Westlake University, leading the AI for Science lab. PhD in physics " +
      "at MIT, then a postdoc at Stanford; works on learning-accelerated simulation, scientific " +
      "discovery, and neuro-symbolic models.",
    group: "westlake",
    href: "https://ai4s.lab.westlake.edu.cn/",
  },
  {
    id: "max-tegmark",
    name: "Max Tegmark",
    affiliation: "Professor of Physics, MIT",
    bio:
      "Professor of Physics at MIT and co-founder of the Future of Life Institute. Works on " +
      "physics-informed machine learning and on the interpretability of neural networks.",
    group: "westlake",
    href: "https://space.mit.edu/home/tegmark/",
  },
  {
    id: "jure-leskovec",
    name: "Jure Leskovec",
    affiliation: "Professor of Computer Science, Stanford",
    bio:
      "Professor of Computer Science at Stanford and a central figure in graph machine learning — " +
      "GraphSAGE, the Open Graph Benchmark, and the SNAP project.",
    group: "westlake",
    href: "https://cs.stanford.edu/people/jure/",
  },
  {
    id: "ziming-liu",
    name: "Ziming Liu",
    affiliation: "Assistant Professor, College of AI, Tsinghua University",
    bio:
      "Assistant Professor in Tsinghua's College of AI and founder of MetaCircle. Works on the " +
      "physics of AI — grokking, interpretability, Kolmogorov-Arnold Networks. Did his PhD with " +
      "Max Tegmark a few years after Tailin Wu, then a postdoc at Stanford: the same path, one " +
      "cohort down.",
    group: "westlake",
    href: "https://kindxiaoming.github.io/",
  },
];

// Author lists are as published; co-authors outside this roster are omitted so
// the graph stays at "my collaborators and their collaborators".
export const papers: Paper[] = [
  {
    id: "sat-mask",
    title: "SAT-Mask: Efficient Diffusion Language Model Training",
    venue: "Under review, 2026",
    authors: ["me", "zhilong-zhang", "yuxuan-song", "keyue-qiu", "hao-zhou"],
  },
  {
    id: "molcraft",
    title: "MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space",
    venue: "ICML 2024",
    authors: [
      "yanru-qu",
      "keyue-qiu",
      "yuxuan-song",
      "jingjing-gong",
      "jiawei-han",
      "mingyue-zheng",
      "hao-zhou",
      "weiying-ma",
    ],
  },
  {
    id: "piloting",
    title: "Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule",
    venue: "ICML 2025",
    authors: [
      "keyue-qiu",
      "yuxuan-song",
      "zhehuan-fan",
      "peidong-liu",
      "zhe-zhang",
      "mingyue-zheng",
      "hao-zhou",
      "weiying-ma",
    ],
  },
  {
    id: "gradient-bfn",
    title: "Empower Structure-Based Molecule Optimization with Gradient Guided Bayesian Flow Networks",
    venue: "ICML 2025",
    authors: [
      "keyue-qiu",
      "yuxuan-song",
      "jie-yu",
      "hongbo-ma",
      "ziyao-cao",
      "zhilong-zhang",
      "yushuai-wu",
      "mingyue-zheng",
      "hao-zhou",
      "weiying-ma",
    ],
  },
  {
    id: "geodesic",
    title: "Demystifying Multimodal Biomolecular Co-design With Intrinsic Geodesic Coupling",
    venue: "ICML 2026",
    authors: ["keyue-qiu", "xintong-wang", "zhilong-zhang", "hao-zhou", "weiying-ma"],
  },
  {
    id: "dcfold",
    title: "DCFold: Efficient Protein Structure Generation with Single Forward Pass",
    venue: "ICLR 2026 (Oral)",
    authors: ["zhe-zhang", "yuanning-feng", "yuxuan-song", "keyue-qiu", "hao-zhou", "weiying-ma"],
  },
  {
    id: "amix2",
    title: "AMix-2: Establishing Protein as a Native Modality in Large Language Models",
    venue: "Technical report, 2026",
    authors: [
      "keyue-qiu",
      "yuxuan-song",
      "lijun-wu",
      "lei-bai",
      "yaqin-zhang",
      "weiying-ma",
      "dahua-lin",
      "bowen-zhou",
      "hao-zhou",
    ],
  },
  {
    id: "seed-diffusion",
    title: "Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference",
    venue: "ByteDance Seed × THU-AIR, 2025",
    authors: ["yuxuan-song", "zheng-zhang", "ge-zhang"],
  },
  // ByteDance AI Lab years, before any of them were at THU-AIR.
  {
    id: "density-ratio",
    title: "Improving Maximum Likelihood Training for Text Generation with Density Ratio Estimation",
    venue: "AISTATS 2020",
    authors: ["yuxuan-song", "ning-miao", "hao-zhou", "lantao-yu", "mingxuan-wang", "lei-li"],
  },
  {
    id: "right-scissors",
    title: "Do You Have the Right Scissors? Tailoring Pre-trained Language Models via Monte-Carlo Methods",
    venue: "ACL 2020",
    authors: ["ning-miao", "yuxuan-song", "hao-zhou", "lei-li"],
  },
  // Zhilong Zhang's master's years at Peking University, in Bin Cui's lab.
  {
    id: "diffusion-survey",
    title: "Diffusion Models: A Comprehensive Survey of Methods and Applications",
    venue: "ACM Computing Surveys, 2023",
    authors: [
      "ling-yang",
      "zhilong-zhang",
      "yang-song",
      "shenda-hong",
      "runsheng-xu",
      "yue-zhao",
      "yingxia-shao",
      "wentao-zhang",
      "bin-cui",
      "minghsuan-yang",
    ],
  },
  {
    id: "context-prediction",
    title: "Improving Diffusion-Based Image Synthesis with Context Prediction",
    venue: "NeurIPS 2023",
    authors: [
      "ling-yang",
      "jingwei-liu",
      "shenda-hong",
      "zhilong-zhang",
      "zhilin-huang",
      "zheming-cai",
      "wentao-zhang",
      "bin-cui",
    ],
  },
  {
    id: "contextdiff",
    title: "Cross-Modal Contextualized Diffusion Models for Text-Guided Visual Generation and Editing",
    venue: "ICLR 2024",
    authors: [
      "ling-yang",
      "zhilong-zhang",
      "zhaochen-yu",
      "jingwei-liu",
      "minkai-xu",
      "stefano-ermon",
      "bin-cui",
    ],
  },
  {
    id: "ipdiff",
    title: "Protein-Ligand Interaction Prior for Binding-Aware 3D Molecule Diffusion Models",
    venue: "ICLR 2024",
    authors: [
      "zhilin-huang",
      "ling-yang",
      "xiangxin-zhou",
      "zhilong-zhang",
      "wentao-zhang",
      "xiawu-zheng",
      "jie-chen",
      "yu-wang",
      "bin-cui",
      "wenming-yang",
    ],
  },
  {
    id: "sadm",
    title: "Structure-Guided Adversarial Training of Diffusion Models",
    venue: "CVPR 2024",
    authors: ["ling-yang", "haotian-qian", "zhilong-zhang", "jingwei-liu", "bin-cui"],
  },
  {
    id: "graphusion",
    title: "Graphusion: Latent Diffusion for Graph Generation",
    venue: "IEEE TKDE, 2024",
    authors: [
      "ling-yang",
      "zhilin-huang",
      "zhilong-zhang",
      "zhongyi-liu",
      "shenda-hong",
      "wentao-zhang",
      "wenming-yang",
      "bin-cui",
      "luxia-zhang",
    ],
  },
  {
    id: "consistency-fm",
    title: "Consistency Flow Matching: Defining Straight Flows with Velocity Consistency",
    venue: "Preprint, 2024",
    authors: [
      "ling-yang",
      "zixiang-zhang",
      "zhilong-zhang",
      "xingchao-liu",
      "minkai-xu",
      "wentao-zhang",
      "chenlin-meng",
      "stefano-ermon",
      "bin-cui",
    ],
  },
  // Minkai Xu's own line of work, from Mila to Stanford.
  {
    id: "geodiff",
    title: "GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation",
    venue: "ICLR 2022",
    authors: ["minkai-xu", "lantao-yu", "yang-song", "chence-shi", "stefano-ermon", "jian-tang"],
  },
  {
    id: "geoldm",
    title: "Geometric Latent Diffusion Models for 3D Molecule Generation",
    venue: "ICML 2023",
    authors: ["minkai-xu", "alexander-powers", "ron-dror", "stefano-ermon", "jure-leskovec"],
  },
];

/**
 * Who advises whom, as `[advisor, student]`.
 *
 * This is the one relation a paper cannot tell you — an author list is a set,
 * with no direction in it — so it has to be stated by hand. Each pair turns
 * the tie the two already share into an arrow pointing from the advisor to
 * the student; every tie left over stays a plain line, meaning co-authorship
 * and nothing more.
 */
export const advising: [string, string][] = [
  ["hao-zhou", "me"],
  ["hao-zhou", "yuxuan-song"],
  ["hao-zhou", "zhilong-zhang"],
  ["hao-zhou", "keyue-qiu"],
  ["weiying-ma", "hao-zhou"],
  ["weiying-ma", "yuxuan-song"],
  ["weiying-ma", "zhilong-zhang"],
  ["weiying-ma", "keyue-qiu"],
  ["yaqin-zhang", "hao-zhou"],
  // CityU. Ning Miao's group sits in the Hong Kong Institute of AI for Science,
  // which Wei-Ying Ma directs; the two share no paper on this map, so the
  // arrow stands on its own.
  ["weiying-ma", "ning-miao"],
  // Peking University.
  ["bin-cui", "ling-yang"],
  ["bin-cui", "wentao-zhang"],
  // Stanford. Ermon and Yang Song share no paper on this map; that arrow
  // stands on its own. Minkai Xu was co-advised, so he gets two.
  ["stefano-ermon", "yang-song"],
  ["stefano-ermon", "minkai-xu"],
  ["jure-leskovec", "minkai-xu"],
  // Mila, before Stanford.
  ["jian-tang", "minkai-xu"],
  // Westlake. None of these rest on a shared paper, which is the point of
  // letting advising stand on its own.
  ["tailin-wu", "me"],
  ["max-tegmark", "tailin-wu"],
  ["jure-leskovec", "tailin-wu"],
  // Ziming Liu hangs off Tegmark rather than off Tailin Wu, because that is the
  // relation there is: a shared adviser, which the hop distance then renders as
  // the two of them standing one step apart.
  ["max-tegmark", "ziming-liu"],
];

/**
 * Ties to leave off the map, as `[personA, personB]`, even though a shared
 * paper produces them. A line has to claim something; sharing one
 * mass-authored report does not always amount to a relationship worth
 * drawing, and no rule derived from the papers can tell which ones do.
 */
export const hiddenTies: [string, string][] = [
  // AMix-2 alone would tie Ya-Qin Zhang to half the map at the weakest weight
  // on it. Being authors 5 and 2 on one technical report is not a relation.
  ["yaqin-zhang", "yuxuan-song"],
  ["yaqin-zhang", "keyue-qiu"],
  // The diffusion survey has ten authors. It is what joins Yang Song to Ling
  // Yang and to Zhilong Zhang, who wrote it with him, and those ties stay; his
  // ties to Bin Cui and Wentao Zhang rest on nothing else, and are not drawn.
  ["yang-song", "bin-cui"],
  ["yang-song", "wentao-zhang"],
  // Likewise Ermon and Wentao Zhang, who share one nine-author preprint — and
  // Minkai Xu and Wentao Zhang, on the same preprint.
  ["stefano-ermon", "wentao-zhang"],
  ["minkai-xu", "wentao-zhang"],
  // GeoDiff joins Jian Tang to Yang Song and to Ermon, once each. Jian Tang is
  // on the map as Minkai Xu's adviser, and that is the one tie he keeps.
  ["jian-tang", "yang-song"],
  ["jian-tang", "stefano-ermon"],
];

/**
 * What a tie says when one of the two people on it is highlighted, as
 * `[personA, personB, label]`. Order does not matter — the label belongs to
 * the tie, so it reads the same whichever end you click. A tie with no entry
 * here stays silent.
 */
export const tieLabels: [string, string, string][] = [
  ["me", "hao-zhou", "Intern adviser @ THU-AIR, 2025"],
  ["me", "tailin-wu", "Intern adviser @ WestlakeU, 2026"],
  ["hao-zhou", "yuxuan-song", "PhD adviser @ THU-AIR"],
  ["hao-zhou", "zhilong-zhang", "PhD adviser @ THU-AIR"],
  ["hao-zhou", "keyue-qiu", "PhD adviser @ THU-AIR"],
  ["weiying-ma", "yuxuan-song", "PhD adviser @ THU-AIR"],
  ["weiying-ma", "zhilong-zhang", "PhD adviser @ THU-AIR"],
  ["weiying-ma", "keyue-qiu", "PhD adviser @ THU-AIR"],
  ["weiying-ma", "hao-zhou", "Boss @ THU-AIR"],
  ["yaqin-zhang", "hao-zhou", "Boss @ THU-AIR"],
  ["weiying-ma", "ning-miao", "Boss @ CityU"],
  ["hao-zhou", "ning-miao", "Colleagues @ ByteDance AI Lab"],
  ["bin-cui", "ling-yang", "PhD adviser @ PKU"],
  ["bin-cui", "wentao-zhang", "PhD adviser @ PKU"],
  ["stefano-ermon", "yang-song", "PhD adviser @ Stanford"],
  ["stefano-ermon", "minkai-xu", "PhD adviser @ Stanford"],
  ["jure-leskovec", "minkai-xu", "PhD adviser @ Stanford"],
  ["jian-tang", "minkai-xu", "MSc adviser @ Mila"],
  ["max-tegmark", "tailin-wu", "PhD adviser @ MIT"],
  ["jure-leskovec", "tailin-wu", "Postdoc adviser @ Stanford"],
  ["max-tegmark", "ziming-liu", "PhD adviser @ MIT"],
];

/**
 * Drop a file at public/assets/img/people/<id>.jpg and it just appears — no
 * code change. An explicit `avatar` still wins, which is how `me` points at
 * the existing prof_pic. Missing files fall through to initials rather than
 * rendering an empty circle.
 */
function resolveAvatar(person: Person): string | undefined {
  if (person.avatar) return person.avatar;
  for (const ext of ["jpg", "jpeg", "png", "webp"]) {
    const relative = `/assets/img/people/${person.id}.${ext}`;
    // Resolve against the project root. Not import.meta.url: Vite rewrites it
    // to the bundled chunk under dist/, which would probe dist/public/.
    if (existsSync(join(process.cwd(), "public", relative))) return relative;
  }
  return undefined;
}

export const roster: Person[] = [me, ...people].map((person) => ({
  ...person,
  avatar: resolveAvatar(person),
}));

const byId = new Map(roster.map((person) => [person.id, person]));

/**
 * Author lists above stay exactly as published. The graph only ever sees the
 * slice of each one that is actually on the map, so narrowing the roster is
 * purely a matter of deleting people — no paper entry needs touching, and
 * putting somebody back restores their ties by itself.
 */
const contributors = new Map(
  papers.map((paper) => [paper.id, paper.authors.filter((author) => byId.has(author))]),
);

export type Edge = {
  a: string;
  b: string;
  weight: number;
  papers: string[];
  /** Set when this tie is an advising one; the id of whoever advises. */
  advisor?: string;
  /** Shown along the tie while either end is highlighted. */
  label?: string;
};

/**
 * Keep only this many strongest ties per person. `Infinity` draws every real
 * co-authorship — correct while the roster is small, see below.
 */
const TIES_PER_PERSON = Infinity;

/**
 * Taking every co-author pair as an edge gives 127 edges on 26 people — an
 * unreadable hairball, and it overstates things: being authors 7 and 9 on a
 * 22-author report is not a relationship.
 *
 * So ties are weighted the standard way (Newman): a shared paper contributes
 * 1/(authors - 1), meaning a focused five-author paper counts for far more
 * than a mass-authored one. Each person then keeps their strongest
 * TIES_PER_PERSON ties and the union is drawn, leaving a backbone where
 * nobody is isolated and everyone stays reachable.
 *
 * That cut-off is off at the moment. On the current seven-person roster there
 * are 18 real pairs out of 21 possible, so there is no hairball to prevent —
 * all the cut-off did was hide the single strongest tie on the map (Wei-Ying
 * Ma and Yuxuan Song, five shared papers). Put it back to 2 or 3 if the
 * roster grows again; the weights are already computed for it.
 *
 * My own ties are exempt. All of them weigh exactly the same (one paper, five
 * authors), so the cut-off would fall in the middle of a tie and silently drop
 * real co-authors — which is indefensible on a page that is my own network.
 */
export const edges: Edge[] = (() => {
  const found = new Map<string, Edge>();
  for (const paper of papers) {
    const sorted = [...contributors.get(paper.id)!].sort();
    if (sorted.length < 2) continue; // only one of them is on the map: no pair to draw
    const contribution = 1 / (sorted.length - 1);
    for (let i = 0; i < sorted.length; i++) {
      for (let j = i + 1; j < sorted.length; j++) {
        const key = `${sorted[i]}|${sorted[j]}`;
        const existing = found.get(key);
        if (existing) {
          existing.weight += contribution;
          existing.papers.push(paper.id);
        } else {
          found.set(key, { a: sorted[i], b: sorted[j], weight: contribution, papers: [paper.id] });
        }
      }
    }
  }

  const hidden = new Set<string>();
  for (const [x, y] of hiddenTies) {
    if (!byId.has(x) || !byId.has(y)) continue;
    const key = [x, y].sort().join("|");
    if (!found.has(key)) throw new Error(`network: "${x}" and "${y}" share no tie to hide`);
    hidden.add(key);
    found.delete(key);
  }

  const perPerson = new Map<string, Edge[]>(roster.map((person) => [person.id, []]));
  for (const edge of found.values()) {
    perPerson.get(edge.a)!.push(edge);
    perPerson.get(edge.b)!.push(edge);
  }

  const kept = new Set<Edge>();
  for (const [id, list] of perPerson) {
    // Deterministic: weight first, then id, so the layout never shifts.
    list.sort((x, y) => y.weight - x.weight || `${x.a}${x.b}`.localeCompare(`${y.a}${y.b}`));
    const limit = id === me.id ? list.length : TIES_PER_PERSON;
    for (const edge of list.slice(0, limit)) kept.add(edge);
  }

  /**
   * The drawn tie between two people, for the hand-written lists above to
   * hang a direction or a caption on. `null` means one of the two is off the
   * roster, which is fine — trimming the roster takes their annotations with
   * it. A pair that is on the map but unconnected is a mistake, and says so
   * rather than quietly dropping what was written about them.
   */
  const tieBetween = (x: string, y: string, claim: string) => {
    if (!byId.has(x) || !byId.has(y)) return null;
    const edge = found.get([x, y].sort().join("|"));
    if (!edge || !kept.has(edge)) {
      throw new Error(`network: ${claim} needs a tie between "${x}" and "${y}", which is not drawn`);
    }
    return edge;
  };

  for (const [advisor, student] of advising) {
    // Somebody trimmed off the roster takes their advising arrow with them.
    if (!byId.has(advisor) || !byId.has(student)) continue;
    const [a, b] = [advisor, student].sort();
    const key = `${a}|${b}`;
    if (hidden.has(key)) {
      throw new Error(`network: "${advisor}" advises "${student}", so their tie cannot also be hidden`);
    }
    // Advising stands on its own. Demanding a shared paper for it would be
    // perverse on a map of who taught whom — an adviser and a student who has
    // not published yet are the plainest relation here.
    const edge = found.get(key) ?? { a, b, weight: 0, papers: [] };
    found.set(key, edge);
    kept.add(edge);
    edge.advisor = advisor;
  }

  for (const [x, y, label] of tieLabels) {
    const edge = tieBetween(x, y, `the label "${label}"`);
    if (edge) edge.label = label;
  }

  return [...kept];
})();

// Guard the invariant the sparsifier already broke once: every person I have
// actually published with must be joined to me on the map.
{
  const linkedToMe = new Set(
    edges.filter((e) => e.a === me.id || e.b === me.id).map((e) => (e.a === me.id ? e.b : e.a)),
  );
  for (const paper of papers) {
    const authors = contributors.get(paper.id)!;
    if (!authors.includes(me.id)) continue;
    for (const author of authors) {
      if (author === me.id || linkedToMe.has(author)) continue;
      throw new Error(`network: "${author}" co-wrote "${paper.id}" with me but has no edge to me`);
    }
  }
}

/** "Yang Song" → "YS"; shown when a person has no avatar image. */
export function initials(name: string) {
  const words = name.split(/[\s-]+/).filter(Boolean);
  if (words.length === 0) return "?";
  return words.slice(0, 2).map((word) => [...word][0] ?? "").join("").toUpperCase();
}

/**
 * Hop distance between every pair of people over the drawn ties: one for a tie
 * you can see, two for a friend of a friend. This is the only thing the layout
 * below is trying to honour.
 */
const hops: Map<string, Map<string, number>> = (() => {
  const near = new Map<string, string[]>(roster.map((person) => [person.id, []]));
  for (const edge of edges) {
    near.get(edge.a)!.push(edge.b);
    near.get(edge.b)!.push(edge.a);
  }
  const all = new Map<string, Map<string, number>>();
  for (const from of roster) {
    const seen = new Map<string, number>([[from.id, 0]]);
    let front = [from.id];
    while (front.length > 0) {
      const next: string[] = [];
      for (const id of front) {
        for (const other of near.get(id)!) {
          if (seen.has(other)) continue;
          seen.set(other, seen.get(id)! + 1);
          next.push(other);
        }
      }
      front = next;
    }
    // Unreachable sits one hop beyond the furthest thing that is reachable.
    const outside = Math.max(...seen.values()) + 1;
    for (const person of roster) if (!seen.has(person.id)) seen.set(person.id, outside);
    all.set(from.id, seen);
  }
  return all;
})();

/**
 * The branches of my network: the connected pieces left when I am taken out of
 * the graph. On the current roster that is AIR on one side and Westlake on the
 * other, joined only through me.
 *
 * They matter because hop count only carries meaning *along* a branch. That
 * Wei-Ying Ma is four hops from Max Tegmark says nothing about either of them,
 * and insisting the drawing reflect it was what bloated the whole picture:
 * every branch had to be pushed radially past every other, so the diameter
 * grew with the sum of their depths. Between branches the only thing worth
 * drawing is that they have nothing to do with each other, which is said by
 * putting them on opposite sides of me.
 */
const branchOf: Map<string, number> = (() => {
  const near = new Map<string, string[]>(roster.map((person) => [person.id, []]));
  for (const edge of edges) {
    if (edge.a === me.id || edge.b === me.id) continue; // I am the join, not a member
    near.get(edge.a)!.push(edge.b);
    near.get(edge.b)!.push(edge.a);
  }
  const mark = new Map<string, number>([[me.id, 0]]);
  let branch = 0;
  for (const person of roster) {
    if (mark.has(person.id)) continue;
    branch += 1;
    const stack = [person.id];
    while (stack.length > 0) {
      const id = stack.pop()!;
      if (mark.has(id)) continue;
      mark.set(id, branch);
      for (const other of near.get(id)!) stack.push(other);
    }
  }
  return mark;
})();

/** Do these two sit on the same branch, or is one of them me? */
function related(a: string, b: string) {
  return a === me.id || b === me.id || branchOf.get(a) === branchOf.get(b);
}

// ── build-time layout ─────────────────────────────────────────────────────
// Stress majorisation (Kamada-Kawai, solved the way Gansner, Koren and North
// do it) rather than a pile of forces.
//
// The whole objective is one sentence: the distance between two people on the
// page should track the number of hops between them on the graph. That single
// statement is everything the picture is asked to say — everyone I work with
// directly sits nearer to me than anyone I only know through them, and the
// same holds for every other person, so Tailin Wu's own adviser lands closer
// to him than my adviser does. People who share nothing are pushed apart
// because their hop count is large, which is what separates Westlake from AIR
// and leaves each of them a cluster. Nothing is assigned by hand.
//
// Tuning a repulsion against a spring against a group-cohesion term cannot
// express that, which is why it kept producing pictures that were centred but
// lumpy, or even but off-centre.

/**
 * Distance one hop is worth. Searched, because pinning it by hand pins it to
 * whichever caption happens to be longest — every tie on the map was 345 long
 * because one label needed 343 — and that is no way to decide the scale of the
 * whole drawing.
 */
const UNIT_TRIES = [345, 305, 270, 390];
/**
 * Each further hop adds this fraction of a unit, keeping the picture compact.
 * The most decisive number in the whole layout — it is what sets whether the
 * rings stay clearly apart — so it is searched rather than chosen.
 */
const FALLOFF_TRIES = [0.3, 0.4, 0.5, 0.65, 0.8, 0.95, 1.1];
/** How much sharing many papers shortens a tie, on top of its hop distance. */
const PAPER_PULL = 0.03;
/**
 * Least angle, at the person they meet at, that two ties are asked to make.
 * Majorisation on its own leaves a clique bunched into a few degrees — honest,
 * since they really are a clique, but it draws as one smudge of near-collinear
 * lines. Opening the fan is the one aesthetic the algorithm does not supply.
 *
 * Asking for too much of it collapses the drawing: on this roster 32° is fine
 * and 36° loses the hop ordering outright. Where that cliff sits depends on
 * the data, so the value is searched rather than chosen — biggest first, and
 * the solver falls back as far as it has to.
 */
const ANGLE_TRIES = [36, 30, 24, 16, 0].map((deg) => (deg * Math.PI) / 180);
/** How much of the shortfall to take back per sweep. Gently: it fights stress. */
const ANGLE_GAIN = 0.32;
/**
 * Daylight to leave between one ring of a person's contacts and the next, and
 * how much of any shortfall to take back per sweep.
 *
 * Until this existed the hop ordering was only ever *asked* for, through the
 * target distances, and asking needed a wide margin between hops to actually
 * come out right — which is exactly what stretched the sparse branches. Tailin
 * Wu's three contacts each wanted to be 2.05 units from one another while
 * sitting one unit from him, which no arrangement of angles allows, so they
 * shoved outwards. Demanding the ordering instead lets the margin be narrow.
 */
const ORDER_MARGIN = 40;
const ORDER_GAIN = 0.5;
/**
 * Clear space a tie must leave around anybody it is not attached to. A line
 * that grazes an avatar reads as though it ended there, which is how Zhilong
 * Zhang's tie to Hao Zhou came to look like a tie to Yuxuan Song.
 */
const TIE_CLEAR = 24;
/**
 * How much of that shortfall to take back per sweep. Gently, and never as a
 * hard constraint: at full strength it tears up the stress solution, which is
 * the thing actually carrying the meaning. Also searched.
 */
const CLEAR_TRIES = [0.3, 0.1, 0];

/**
 * Starting angles to try. Majorisation is a local method, so where it begins
 * decides which arrangement it finds: on this roster ten starts landed in only
 * three distinct drawings, one of them far tidier than the other two. Which
 * start is the lucky one is not predictable, so they are all tried.
 */
const STARTS = ["", "a", "b", "c", "d", "e", "f", "g"];
/** Sweeps of the majorisation for the drawing that gets kept. */
const ROUNDS = 400;
/**
 * Sweeps while searching. A quarter of the work ranks candidates almost as
 * well, so the grid can be wide; the shortlist is then redone in full.
 */
const SEARCH_ROUNDS = 60;
/** How many of the best coarse candidates to settle properly. */
const SHORTLIST = 16;
/** Advance width of one label character: the label font is mono at 11px. */
const LABEL_CHAR = 6.6;
/** Clear space a label wants beyond the text and the two avatars it spans. */
const LABEL_ROOM = 60;
const NODE_R = 30;
const HUB_R = 42;
const MIN_SEPARATION = 34;

export type Placed = Person & { x: number; y: number; r: number };

/**
 * How long a tie has to be to print its own caption in the gap between the
 * two avatars. A short tie with a long caption lays the words across somebody's
 * face, so this becomes a floor under the distance the two are placed at.
 */
function captionRoom(edge: Edge, ra: number, rb: number) {
  if (!edge.label) return 0;
  return edge.label.length * LABEL_CHAR + ra + rb + LABEL_ROOM;
}

/**
 * How thick a tie is drawn, and how much it shortens: both read off the count
 * of papers the two share. Square-rooted — it is the step from one paper to
 * two that means something, not the step from six to seven.
 *
 * A tie that exists on advising alone has no papers under it and is treated as
 * though it had one, rather than being drawn at zero width.
 */
function tieScale(edge: Edge) {
  return Math.sqrt(Math.max(1, edge.papers.length));
}

/** Stroke width for a tie, on that same scale. One shared paper is hairline. */
export function tieWidth(edge: Edge) {
  return Math.round(tieScale(edge) * 100) / 100;
}

function seed(key: string) {
  let h = 2166136261;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967296;
}

type Tune = { salt: string; angleMin: number; clearGain: number; falloff: number; unit: number };

/**
 * Where everybody starts: on a radial tree. Breadth-first from me, each of my
 * direct contacts is given a slice of the circle in proportion to how many
 * people hang off them, their own contacts a slice of that slice, and so on
 * outwards. Majorisation is a local method and keeps whatever arrangement it
 * is handed: started from people scattered at random angles it left the
 * Peking clique strung out around me and me at the edge of the picture. A
 * tree start hands it the drawing it is looking for — me in the middle, each
 * branch in a sector of its own — and leaves it only the tidying. The salt
 * decides the order children take round the circle, and where it begins.
 *
 * Returns each person's bearing from me. Anyone the tree cannot reach is
 * left out, and takes a random bearing instead.
 */
function radialStart(salt: string): Map<string, number> {
  const near = new Map<string, string[]>(roster.map((person) => [person.id, []]));
  for (const edge of edges) {
    near.get(edge.a)!.push(edge.b);
    near.get(edge.b)!.push(edge.a);
  }
  const children = new Map<string, string[]>(roster.map((person) => [person.id, []]));
  const claimed = new Set<string>([me.id]);
  let front = [me.id];
  while (front.length > 0) {
    const next: string[] = [];
    for (const id of front) {
      const kids = near
        .get(id)!
        .filter((other) => !claimed.has(other))
        .sort((x, y) => seed(x + salt) - seed(y + salt));
      for (const kid of kids) {
        claimed.add(kid);
        children.get(id)!.push(kid);
        next.push(kid);
      }
    }
    front = next;
  }
  // How many leaves hang off each person: the share of the circle they need.
  const leaves = new Map<string, number>();
  const count = (id: string): number => {
    const kids = children.get(id)!;
    const n = kids.length === 0 ? 1 : kids.reduce((sum, kid) => sum + count(kid), 0);
    leaves.set(id, n);
    return n;
  };
  count(me.id);
  const bearing = new Map<string, number>();
  const spread = (id: string, from: number, to: number) => {
    let at = from;
    for (const kid of children.get(id)!) {
      const span = ((to - from) * leaves.get(kid)!) / leaves.get(id)!;
      bearing.set(kid, at + span / 2);
      spread(kid, at, at + span);
      at += span;
    }
  };
  const turn = seed(salt) * Math.PI * 2;
  spread(me.id, turn, turn + Math.PI * 2);
  return bearing;
}

/**
 * Two drawings of the same graph come out of the one solver.
 *
 * The open one (`captions: true`) leaves every labelled tie long enough to
 * print its caption between the two faces, and is what a click on a person
 * unfolds into. The folded one ignores captions altogether: it is built from
 * the relations alone and is what [ universe ] shows. Solving the folded one
 * *from* the open one (`from`) is what makes the unfolding read as one picture
 * opening rather than two pictures swapped — every person slides in or out
 * along much the same line, and the drawing keeps its orientation.
 */
type Mode = { captions: boolean; from?: { place: Placed[]; unit: number } };

const OPEN: Mode = { captions: true };

function layout({ salt, angleMin, clearGain, falloff, unit }: Tune, rounds: number, mode: Mode = OPEN): Placed[] {
  const start = new Map(mode.from?.place.map((person) => [person.id, person]));
  const shrink = mode.from ? unit / mode.from.unit : 1;
  const bearings = mode.from ? new Map<string, number>() : radialStart(salt);
  const nodes = roster.map((person) => {
    const angle = bearings.get(person.id) ?? seed(person.id + salt) * Math.PI * 2;
    const hop = hops.get(me.id)!.get(person.id)!;
    const out = unit * (1 + falloff * (hop - 1));
    const from = start.get(person.id);
    return {
      person,
      r: person.id === me.id ? HUB_R : NODE_R,
      x: person.id === me.id ? 0 : from ? from.x * shrink : Math.cos(angle) * out,
      y: person.id === me.id ? 0 : from ? from.y * shrink : Math.sin(angle) * out,
      // I stay at the origin, so that the picture is centred on me by
      // construction rather than by luck.
      pinned: person.id === me.id,
    };
  });

  type Body = (typeof nodes)[number];

  const byPair = new Map<string, Edge>();
  for (const edge of edges) byPair.set([edge.a, edge.b].sort().join("|"), edge);

  /** Room a tie must leave for its caption — none at all in the folded drawing. */
  function room(edge: Edge, ra: number, rb: number) {
    return mode.captions ? captionRoom(edge, ra, rb) : 0;
  }

  /** How far from me a person's hop count puts them. */
  function radius(node: Body) {
    return unit * (1 + falloff * (hops.get(me.id)!.get(node.person.id)! - 1));
  }

  /** What we would like the distance between two people to be. */
  function wanted(a: Body, b: Body) {
    // Nothing to do with each other: the most the drawing can say is that, and
    // it says it by standing them on opposite sides of me.
    if (!related(a.person.id, b.person.id)) {
      return Math.max(radius(a) + radius(b), a.r + b.r + MIN_SEPARATION);
    }
    const hop = hops.get(a.person.id)!.get(b.person.id)!;
    let want = unit * (1 + falloff * (hop - 1));
    const edge = byPair.get([a.person.id, b.person.id].sort().join("|"));
    if (edge) {
      want *= 1 - PAPER_PULL * (tieScale(edge) - 1);
      want = Math.max(want, room(edge, a.r, b.r));
    }
    // Nobody may be asked to stand inside somebody else.
    return Math.max(want, a.r + b.r + MIN_SEPARATION);
  }

  // Resolve the target distances and their weights once. Weighting by 1/d²
  // is the usual choice: it asks for the short distances to come out right,
  // which is where the reader actually looks.
  const goal = nodes.map((a) =>
    nodes.map((b) => {
      if (a === b) return { want: 0, weight: 0 };
      const want = wanted(a, b);
      return { want, weight: 1 / (want * want) };
    }),
  );

  /** Push two bodies apart to `want`. A pinned one stays; the other gives. */
  function separate(a: Body, b: Body, want: number) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const d = Math.hypot(dx, dy) || 1;
    if (d >= want) return;
    const shift = (want - d) / d;
    const share = a.pinned ? 0 : b.pinned ? 1 : 0.5;
    a.x -= dx * shift * share;
    a.y -= dy * shift * share;
    b.x += dx * shift * (1 - share);
    b.y += dy * shift * (1 - share);
  }

  const ends = new Map(nodes.map((node) => [node.person.id, node]));
  const fans = nodes.map((node) => ({
    node,
    near: edges
      .filter((edge) => edge.a === node.person.id || edge.b === node.person.id)
      .map((edge) => ends.get(edge.a === node.person.id ? edge.b : edge.a)!),
  }));

  /** Turn a body about a pivot. Distance to the pivot is untouched. */
  function spin(node: Body, about: Body, by: number) {
    if (node.pinned) return;
    const dx = node.x - about.x;
    const dy = node.y - about.y;
    const d = Math.hypot(dx, dy);
    const at = Math.atan2(dy, dx) + by;
    node.x = about.x + Math.cos(at) * d;
    node.y = about.y + Math.sin(at) * d;
  }

  // My view only: see orderRings.
  const views = nodes.filter((node) => node.pinned).map((node) => {
    const seen = hops.get(node.person.id)!;
    const others = nodes.filter((other) => other !== node);
    const levels = [...new Set(others.map((other) => seen.get(other.person.id)!))].sort(
      (x, y) => x - y,
    );
    return {
      node,
      // Everyone else, grouped by how many hops from this person they are.
      rings: levels.map((hop) => others.filter((other) => seen.get(other.person.id) === hop)),
    };
  });

  /**
   * Insist on the one thing the target distances only ask for politely: from
   * my point of view, whoever I work with directly is nearer than whoever I
   * only know through somebody else, ring by ring outwards. Worked radially
   * about me, so it changes how far off somebody is and never which direction
   * they lie in.
   *
   * It used to be asked of everybody's point of view, and could be while the
   * map was two branches meeting at me. Once Jure Leskovec joined the Westlake
   * side to the Peking side the map became one cyclic piece with an
   * eleven-way clique in it, and the demand became unsatisfiable in a plane:
   * every person's push raised the floor for the next, round on round, until
   * the coordinates overflowed. I am pinned, so pushes from my view cannot
   * cascade. Everybody else's ordering is left to the target distances.
   */
  function orderRings() {
    for (const { node, rings } of views) {
      let floor = 0;
      for (const ring of rings) {
        let outermost = floor;
        for (const other of ring) {
          const dx = other.x - node.x;
          const dy = other.y - node.y;
          const d = Math.hypot(dx, dy) || 1;
          let out = d;
          if (d < floor && !other.pinned) {
            out = d + (floor - d) * ORDER_GAIN;
            other.x = node.x + (dx / d) * out;
            other.y = node.y + (dy / d) * out;
          }
          outermost = Math.max(outermost, out);
        }
        floor = outermost + ORDER_MARGIN;
      }
    }
  }

  /**
   * Open up the angle between every pair of ties meeting at one person. Each
   * neighbour only rotates about that person, so its distance to them — the
   * thing carrying the hop ordering — comes through unchanged.
   */
  function openFans() {
    for (const { node, near } of fans) {
      for (let i = 0; i < near.length; i++) {
        for (let j = i + 1; j < near.length; j++) {
          const u = near[i];
          const v = near[j];
          const au = Math.atan2(u.y - node.y, u.x - node.x);
          const av = Math.atan2(v.y - node.y, v.x - node.x);
          let delta = av - au;
          while (delta > Math.PI) delta -= 2 * Math.PI;
          while (delta < -Math.PI) delta += 2 * Math.PI;
          const gap = Math.abs(delta);
          if (gap >= angleMin) continue;
          const turn = ((angleMin - gap) / 2) * Math.sign(delta || 1) * ANGLE_GAIN;
          spin(u, node, -turn);
          spin(v, node, turn);
        }
      }
    }
  }

  const spans = edges.map((edge) => ({
    edge,
    a: ends.get(edge.a)!,
    b: ends.get(edge.b)!,
    // I am left out as an obstacle. I sit at the centre of my own clique, so
    // its chords pass near me whatever is done, and my avatar is opaque, so a
    // tie behind me is hidden rather than ambiguous.
    off: nodes.filter(
      (node) => !node.pinned && node.person.id !== edge.a && node.person.id !== edge.b,
    ),
  }));

  /** Move a body, unless it is pinned. */
  function shove(node: Body, dx: number, dy: number) {
    if (node.pinned) return;
    node.x += dx;
    node.y += dy;
  }

  /** Push a person off any tie they have nothing to do with. */
  function clearOfTies() {
    for (const { a, b, off } of spans) {
      for (const node of off) {
        const vx = b.x - a.x;
        const vy = b.y - a.y;
        const len2 = vx * vx + vy * vy || 1;
        // Where along the tie the person is nearest, clamped to its ends.
        const t = Math.max(0, Math.min(1, ((node.x - a.x) * vx + (node.y - a.y) * vy) / len2));
        let dx = node.x - (a.x + t * vx);
        let dy = node.y - (a.y + t * vy);
        let d = Math.hypot(dx, dy);
        const want = node.r + TIE_CLEAR;
        if (d >= want) continue;
        if (d < 1e-6) {
          // Dead on the line, so there is no side to be on: pick one.
          dx = -vy;
          dy = vx;
          d = Math.hypot(dx, dy) || 1;
        }
        const push = ((want - d) / d) * clearGain;
        // The person steps aside and the tie bends the other way, each end
        // giving in proportion to how near the touch is to it.
        const mine = node.pinned ? 0 : 0.5;
        shove(node, dx * push * mine, dy * push * mine);
        const theirs = (1 - mine) * push;
        shove(a, -dx * theirs * (1 - t), -dy * theirs * (1 - t));
        shove(b, -dx * theirs * t, -dy * theirs * t);
      }
    }
  }

  for (let round = 0; round < rounds; round++) {
    // One majorisation sweep. Each person moves to the weighted average of
    // where everybody else would like them to be, which is guaranteed not to
    // make the overall stress worse.
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      if (node.pinned) continue;
      let sx = 0;
      let sy = 0;
      let sw = 0;
      for (let j = 0; j < nodes.length; j++) {
        if (i === j) continue;
        const other = nodes[j];
        const { want, weight } = goal[i][j];
        const dx = node.x - other.x;
        const dy = node.y - other.y;
        const d = Math.hypot(dx, dy) || 1;
        sx += weight * (other.x + (want * dx) / d);
        sy += weight * (other.y + (want * dy) / d);
        sw += weight;
      }
      node.x = sx / sw;
      node.y = sy / sw;
    }

    openFans();
    clearOfTies();

    // The things that are not up for negotiation.
    for (let pass = 0; pass < 2; pass++) {
      for (const edge of edges) {
        const a = nodes.find((node) => node.person.id === edge.a)!;
        const b = nodes.find((node) => node.person.id === edge.b)!;
        separate(a, b, room(edge, a.r, b.r));
      }
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          separate(nodes[i], nodes[j], nodes[i].r + nodes[j].r + MIN_SEPARATION);
        }
      }
      // Last word, so that pushing people apart cannot reshuffle the rings.
      orderRings();
    }
  }

  // Stress depends only on distances, so the drawing is free to be turned.
  // Turn its long axis flat: the canvas is far wider than it is tall, and a
  // rotation is the one way to exploit that without distorting a thing.
  //
  // A drawing folded from another is turned to lie over that one instead —
  // the rotation about me that leaves everybody, summed, nearest to where the
  // open drawing has them — so that the page can slide between the two with
  // each person moving in or out along a line, not round a bend.
  let turn: number;
  if (mode.from) {
    let sin = 0;
    let cos = 0;
    for (const node of nodes) {
      const was = start.get(node.person.id)!;
      cos += node.x * was.x + node.y * was.y;
      sin += node.x * was.y - node.y * was.x;
    }
    turn = Math.atan2(sin, cos);
  } else {
    let sxx = 0;
    let sxy = 0;
    let syy = 0;
    for (const node of nodes) {
      sxx += node.x * node.x;
      sxy += node.x * node.y;
      syy += node.y * node.y;
    }
    turn = -0.5 * Math.atan2(2 * sxy, sxx - syy);
  }
  const ct = Math.cos(turn);
  const st = Math.sin(turn);

  const round = (value: number) => Math.round(value * 10) / 10;
  return nodes.map((node) => ({
    ...node.person,
    x: round(node.x * ct - node.y * st),
    y: round(node.x * st + node.y * ct),
    r: node.r,
  }));
}

/**
 * What a drawing is judged on, most important first. Writing the order down is
 * the whole point of this block: these goals pull against each other, and every
 * round of hand-tuning came down to deciding which one was allowed to give. Now
 * the solver decides, the same way, every time the roster changes.
 *
 *  1. NOTHING FALSE. Nobody may overlap anybody, and no tie may cross a face it
 *     has nothing to do with. An avatar is opaque, so a tie passing over one is
 *     cut in two and reads as two different ties — it invents a relationship
 *     that does not exist. This is the only tier that can veto a drawing.
 *  2. DISTANCE FROM ME MEANS HOPS FROM ME: whoever I work with directly sits
 *     nearer to me than whoever I only know through somebody else, ring by
 *     ring. This is what the picture is for. It was asked of everybody's
 *     view while the map was two branches meeting at me; now that it is one
 *     cyclic piece that cannot be had in a plane (see orderRings), so for
 *     everybody else the target distances ask for it without insisting.
 *  3. THE DRAWING IS COMPACT. An oversized drawing is what "why are the lines
 *     over here so long" turns out to mean. This used to be a floor — how much
 *     of the drawing fit the frame was what set the captions' size on screen —
 *     but the page no longer shows the open drawing whole: [ universe ] shows
 *     the folded one, and a click focuses at a zoom the captions read at. So
 *     it is a cost now, weighed above composition but able to give way; on a
 *     roster of seventeen the whole drawing simply does not fit at 0.8.
 *  4. I AM IN THE MIDDLE.
 *  5. TIES COME OUT SIMILAR LENGTHS. Every tie is one hop, so if one is twice
 *     another the length has stopped meaning anything. It cannot be had
 *     outright — a clique cannot hold its distances in a plane, so its ties
 *     squash, while a tree holds them exactly and looks stretched next to it —
 *     but it is worth weighing, and until now the solver could not see it at
 *     all.
 *  6. TIES FAN OUT where they meet, rather than lying along each other.
 *  7. FEW CROSSINGS. Last — and not really a choice: my own co-authors form a
 *     K5, and Wei-Ying Ma turns it into K6 minus an edge, so two crossings are
 *     unavoidable however the thing is drawn. Chasing them is not worth giving
 *     up anything above.
 *
 * Judged in three steps, not one weighted sum. Hand-weighting six goals against
 * each other turned into whack-a-mole: raising the price of legibility bought a
 * readable drawing whose lines lay 11° apart and grazed people's faces, which
 * is what the weights before it had been raised to stop.
 *
 *   WRONG   tiers 1 and 2, counted. Any of it vetoes the drawing outright.
 *   SHORT   floors below which a drawing stops being legible at all — set at
 *           levels already achieved, so they are known to be reachable, and
 *           counted rather than priced so that nothing can be traded for them.
 *   COST    what is genuinely a matter of taste: where I sit, how even the
 *           ties are, how much room there is above each floor, crossings.
 */
/** Least daylight a tie may leave around a face it passes. */
const DAYLIGHT_MIN = 20;
/** Least angle any two ties meeting at a person may make. */
const FAN_MIN = (14 * Math.PI) / 180;
/** Above the floors it is only taste, so these are small and only break ties. */
const CENTRE_COST = 3;
const EVEN_COST = 2.5;
const ZOOM_COST = 1.5;
const ANGLE_COST = 1;
const CROSSING_COST = 0.5;
/**
 * Room above the daylight floor, priced like the room above the fan floor.
 * Until this existed the cost saw no difference between a tie one unit off a
 * face and one nineteen off it, so when no candidate cleared the floor — as
 * none does at nineteen people — the folded drawing came back with a line
 * touching a face, which is the one thing a fold must not do.
 */
const DAYLIGHT_COST = 1.5;
/**
 * The desktop stage the drawing is judged against, in the page's units of one
 * CSS pixel each. The page itself sizes its viewBox to whatever stage it gets,
 * a phone's included; legibility there is the page's business, not the layout's.
 */
const VIEW_W = 1240;
const VIEW_H = 720;

type Grade = {
  /** Tiers 1 and 2: anything here makes the drawing say something untrue. */
  faults: number;
  /** Legibility floors missed. Counted, so nothing can be traded for them. */
  short: number;
  /** Narrowest space between a tie and a face it passes. Negative is a fault. */
  daylight: number;
  /** Tiers 3 to 6, priced and summed. */
  cost: number;
  offCentre: number;
  /** Spread of tie lengths, as a coefficient of variation. */
  uneven: number;
  narrowest: number;
  zoom: number;
  crossings: number;
};

/**
 * `behindMe` lets a tie pass behind my own face without penalty, on the same
 * grounds `clearOfTies` already leaves me out as an obstacle: I sit at the
 * centre of my own clique, so packed tightly its chords cannot help but cross
 * me, and since everyone on such a chord is joined to me anyway, a line
 * disappearing behind my avatar invents nothing. The open drawing is judged
 * without it, as it always was.
 */
function grade(place: Placed[], { behindMe = false } = {}): Grade {
  const at = new Map(place.map((person) => [person.id, person]));
  const gap = (a: Placed, b: Placed) => Math.hypot(a.x - b.x, a.y - b.y);

  // ── tier 1 ────────────────────────────────────────────────────────────────
  let faults = 0;
  for (let i = 0; i < place.length; i++) {
    for (let j = i + 1; j < place.length; j++) {
      if (gap(place[i], place[j]) < place[i].r + place[j].r) faults += 1;
    }
  }
  let daylight = Infinity;
  for (const edge of edges) {
    const a = at.get(edge.a)!;
    const b = at.get(edge.b)!;
    for (const other of place) {
      if (other.id === edge.a || other.id === edge.b) continue;
      if (behindMe && other.id === me.id) continue;
      const vx = b.x - a.x;
      const vy = b.y - a.y;
      const len2 = vx * vx + vy * vy || 1;
      const t = Math.max(0, Math.min(1, ((other.x - a.x) * vx + (other.y - a.y) * vy) / len2));
      const clear = Math.hypot(a.x + t * vx - other.x, a.y + t * vy - other.y) - other.r;
      daylight = Math.min(daylight, clear);
      if (clear < 0) faults += 1;
    }
  }

  // ── tier 2 ────────────────────────────────────────────────────────────────
  // From my view only, for the reason given at orderRings.
  for (const person of place.filter((person) => person.id === me.id)) {
    const rings = new Map<number, { near: number; far: number }>();
    for (const other of place) {
      if (other.id === person.id) continue;
      // Only along their own branch, plus me: see `branchOf`. Ordering a person
      // against a branch they have no connection to is meaningless, and
      // demanding it is what stretched every branch past every other.
      if (!related(person.id, other.id)) continue;
      const hop = hops.get(person.id)!.get(other.id)!;
      const d = gap(person, other);
      const ring = rings.get(hop);
      if (ring) {
        ring.near = Math.min(ring.near, d);
        ring.far = Math.max(ring.far, d);
      } else {
        rings.set(hop, { near: d, far: d });
      }
    }
    const levels = [...rings.keys()].sort((x, y) => x - y);
    for (let i = 0; i + 1 < levels.length; i++) {
      if (rings.get(levels[i + 1])!.near <= rings.get(levels[i])!.far) faults += 1;
    }
  }

  // ── tiers 3 to 6 ──────────────────────────────────────────────────────────
  const xs = place.map((person) => person.x);
  const ys = place.map((person) => person.y);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  const mid = { x: (Math.min(...xs) + Math.max(...xs)) / 2, y: (Math.min(...ys) + Math.max(...ys)) / 2 };
  const hub = at.get(me.id)!;
  const offCentre = Math.hypot(hub.x - mid.x, hub.y - mid.y) / (Math.max(width, height) || 1);

  const lengths = edges.map((edge) => gap(at.get(edge.a)!, at.get(edge.b)!));
  const mean = lengths.reduce((sum, len) => sum + len, 0) / (lengths.length || 1);
  const uneven =
    mean > 0
      ? Math.sqrt(lengths.reduce((sum, len) => sum + (len - mean) ** 2, 0) / lengths.length) / mean
      : 0;

  let narrowest = Math.PI * 2;
  for (const person of place) {
    const bearings = edges
      .filter((edge) => edge.a === person.id || edge.b === person.id)
      .map((edge) => {
        const other = at.get(edge.a === person.id ? edge.b : edge.a)!;
        return Math.atan2(other.y - person.y, other.x - person.x);
      })
      .sort((x, y) => x - y);
    if (bearings.length < 2) continue;
    for (let i = 0; i < bearings.length; i++) {
      const next = i + 1 < bearings.length ? bearings[i + 1] : bearings[0] + Math.PI * 2;
      narrowest = Math.min(narrowest, next - bearings[i]);
    }
  }

  // The page fits the whole drawing into the viewBox; this is that sum.
  const zoom = Math.min(2.5, Math.max(0.3, Math.min(VIEW_W / (width + 120), VIEW_H / (height + 140))));

  const straddles = (p: Placed, q: Placed, r: Placed, t: Placed) => {
    const side = (o: Placed, x: Placed, y: Placed) =>
      (x.x - o.x) * (y.y - o.y) - (x.y - o.y) * (y.x - o.x);
    return (
      side(r, t, p) > 0 !== side(r, t, q) > 0 && side(p, q, r) > 0 !== side(p, q, t) > 0
    );
  };
  let crossings = 0;
  for (let i = 0; i < edges.length; i++) {
    for (let j = i + 1; j < edges.length; j++) {
      const one = edges[i];
      const two = edges[j];
      if (new Set([one.a, one.b, two.a, two.b]).size < 4) continue;
      if (straddles(at.get(one.a)!, at.get(one.b)!, at.get(two.a)!, at.get(two.b)!)) crossings += 1;
    }
  }

  const cost =
    ZOOM_COST * Math.max(0, 1 - zoom) +
    CENTRE_COST * offCentre +
    EVEN_COST * uneven +
      ANGLE_COST * Math.max(0, 1 - narrowest / (Math.PI / 6)) +
    DAYLIGHT_COST * Math.max(0, 1 - daylight / DAYLIGHT_MIN) +
    (CROSSING_COST * crossings) / Math.max(1, edges.length);

  const short = (daylight < DAYLIGHT_MIN ? 1 : 0) + (narrowest < FAN_MIN ? 1 : 0);

  return { faults, short, daylight, cost, offCentre, uneven, narrowest, zoom, crossings };
}

/**
 * Try every tuning and keep the best drawing. Deterministic, so the page is
 * identical on every build, and self-correcting: adding a person re-runs the
 * whole search instead of leaving the old hand-picked numbers to rot.
 */
/** Wrong, then short of a floor, then cost. Never traded against each other. */
function beats(mark: Grade, than: Grade) {
  if (mark.faults !== than.faults) return mark.faults < than.faults;
  if (mark.short !== than.short) return mark.short < than.short;
  return mark.cost < than.cost;
}

function solve(): { place: Placed[]; tune: Tune } {
  const grid: Tune[] = [];
  for (const salt of STARTS) {
    for (const angleMin of ANGLE_TRIES) {
      for (const clearGain of CLEAR_TRIES) {
        for (const falloff of FALLOFF_TRIES) {
          for (const unit of UNIT_TRIES) grid.push({ salt, angleMin, clearGain, falloff, unit });
        }
      }
    }
  }

  // Many tunings land on the same drawing, so a plain top-16 was fifteen
  // copies of one candidate and one alternative. Keep them distinct.
  const seen = new Set<string>();
  const shortlist = grid
    .map((tune) => ({ tune, mark: grade(layout(tune, SEARCH_ROUNDS)) }))
    .sort((x, y) => (beats(x.mark, y.mark) ? -1 : 1))
    .filter(({ mark }) => {
      const shape = `${mark.faults}|${mark.short}|${mark.uneven.toFixed(2)}|${mark.zoom.toFixed(2)}|${mark.crossings}`;
      if (seen.has(shape)) return false;
      seen.add(shape);
      return true;
    })
    .slice(0, SHORTLIST);

  let best: { place: Placed[]; mark: Grade; tune: Tune } | null = null;
  for (const { tune } of shortlist) {
    const place = layout(tune, ROUNDS);
    const mark = grade(place);
    if (!best || beats(mark, best.mark)) best = { place, mark, tune };
  }

  const { mark, tune } = best!;
  const tried = grid.length;
  const deg = (radians: number) => `${Math.round((radians * 180) / Math.PI)}\u00b0`;
  // Printed rather than checked by a script: adding a person reruns this, and
  // the numbers that mattered while tuning it are the numbers worth seeing.
  console.log(
    [
      `[network] ${roster.length} people, ${edges.length} ties, best of ${tried} candidates`,
      `  faults ${mark.faults}${mark.faults === 0 ? " (nothing false)" : " \u2605 SOMETHING IS WRONG"}`,
      `  floors missed ${mark.short}${mark.short === 0 ? " (all legible)" : " \u2605 BELOW A FLOOR"}`,
      `  hops honoured from my view: ${mark.faults === 0 ? "yes" : "no"}`,
      `  I am ${(mark.offCentre * 100).toFixed(1)}% off centre`,
      `  tie lengths vary by ${(mark.uneven * 100).toFixed(0)}%`,
      `  narrowest fan ${deg(mark.narrowest)}, ties clear faces by ${mark.daylight.toFixed(0)}`,
      `  zoom ${mark.zoom.toFixed(2)}, crossings ${mark.crossings}, cost ${mark.cost.toFixed(3)}`,
      `  won with start "${tune.salt}", fan ${deg(tune.angleMin)}, clearance ${tune.clearGain},` +
        ` falloff ${tune.falloff}, unit ${tune.unit}`,
    ].join("\n"),
  );
  return best!;
}

/**
 * Hop lengths to try for the folded drawing, smallest first. The search stops
 * at the first that says nothing false and clears every legibility floor: the
 * folded drawing exists to be small, so nothing above the floors is weighed
 * against that.
 *
 * On this roster the AIR clique is what sets the floor. Seven faces of radius
 * 30 joined nearly every way round cannot be packed much under a hop of 240
 * without some chord grazing a face — Keyue Qiu to Wei-Ying Ma passed eight
 * units from Hao Zhou at 150 — and a line that grazes a face reads as ending
 * there. That is a fact about the graph, not a tuning to be argued with. The
 * list runs up close to the open hop so that there is always something to
 * fall back to; the build log says where it landed.
 */
const FOLD_UNIT_TRIES = [150, 170, 190, 210, 225, 240, 260, 280, 300];

/**
 * The folded drawing: the open one re-solved with every caption floor taken
 * away, at a much shorter hop, started from where the open one left everybody.
 * Only the start is inherited from the open drawing's winning tune. Hop length
 * and falloff are searched again, since those are what the caption floors had
 * been forcing; so are clearance and fan, because what wins at a long hop —
 * no clearance at all, say, since long ties pass nobody closely — is not what
 * a packed drawing needs.
 *
 * Searched the way the open drawing is: every tuning at a hop is settled
 * coarsely, the best few are redone in full, and the hop is accepted at the
 * first that says nothing false and clears every floor.
 */
function fold({ place, tune }: { place: Placed[]; tune: Tune }): Placed[] {
  const from = { place, unit: tune.unit };
  const mode = { captions: false, from };
  let best: { place: Placed[]; mark: Grade; unit: number } | null = null;
  // A fold must be a fold: only hops shorter than the open drawing's.
  for (const unit of FOLD_UNIT_TRIES.filter((unit) => unit < tune.unit)) {
    const grid: Tune[] = [];
    for (const falloff of FALLOFF_TRIES) {
      for (const clearGain of CLEAR_TRIES) {
        for (const angleMin of ANGLE_TRIES) grid.push({ ...tune, unit, falloff, clearGain, angleMin });
      }
    }
    const shortlist = grid
      .map((t) => ({ tune: t, mark: grade(layout(t, SEARCH_ROUNDS, mode), { behindMe: true }) }))
      .sort((x, y) => (beats(x.mark, y.mark) ? -1 : 1))
      .slice(0, SHORTLIST / 2);
    for (const { tune: t } of shortlist) {
      const candidate = layout(t, ROUNDS, mode);
      const mark = grade(candidate, { behindMe: true });
      if (!best || beats(mark, best.mark)) best = { place: candidate, mark, unit };
    }
    if (best!.mark.faults === 0 && best!.mark.short === 0) break;
  }
  const { mark, unit } = best!;
  console.log(
    `[network] folded at unit ${unit}: faults ${mark.faults}, floors missed ${mark.short},` +
      ` narrowest fan ${Math.round((mark.narrowest * 180) / Math.PI)}°, ties clear faces by` +
      ` ${mark.daylight.toFixed(0)}, tie lengths vary by ${(mark.uneven * 100).toFixed(0)}%,` +
      ` crossings ${mark.crossings}`,
  );
  return best!.place;
}

const open = solve();

/** The open drawing: room for every caption. What a click on a person unfolds into. */
export const placed: Placed[] = open.place;

/** The folded drawing: relations only, no captions. What [ universe ] shows. */
export const compact: Placed[] = fold(open);
