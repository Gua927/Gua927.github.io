// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-掩码扩散模型中的掩码调度",
        
          title: "掩码扩散模型中的掩码调度",
        
        description: "在扩散模型（Diffusion Models）尤其是掩码扩散模型（Mask Diffusion Models, MDM）的研究中，Noise Scheduler 的设计是决定模型学习上限与采样质量的核心枢纽。本文对 MDM 中的 Noise Scheduler 进行了全面而深入的统一分析，系统性地探讨了当前主流的调度策略，通过对比分析各策略的优缺点，本文揭示了精细化调度策略如何重塑 Token 间的依赖动力学，并在最后对未来突破底层逻辑缺陷的发展方向进行了展望。",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/masking-schedulers-cn/";
          
        },
      },{id: "post-masking-schedulers-of-mask-diffusion-model",
        
          title: "Masking Schedulers of Mask Diffusion Model",
        
        description: "In Mask Diffusion Models (MDM), the Noise Scheduler is pivotal for learning capacity and sampling quality. This paper presents a unified analysis addressing three core challenges —— Exposure Bias induced by Absorb mechanisms, efficiency bottlenecks from Intrinsic Order, and joint probability deviations from Independence Assumptions. We systematically review mainstream strategies, comparing their efficacy in semantic capture, remasking, and efficiency to elucidate how refined scheduling reshapes token dependencies. Finally, we outline future directions for overcoming these underlying logical defects.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/Note-Masking-Schedulers-of-Mask-Diffusion-Model-EN/";
          
        },
      },{id: "post-flow-matching-and-continuous-normalizing-flows",
        
          title: "Flow Matching and Continuous Normalizing Flows",
        
        description: "This post explores Flow-based Models, Continuous Normalizing Flows (CNFs), and Flow Matching (FM). We discuss Normalizing Flows, derive the conditional flow matching objective, and examine special instances including diffusion models and optimal transport.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/Note-FM/";
          
        },
      },{id: "post-the-unification-of-ddpm-and-score-based-models",
        
          title: "The Unification of DDPM and Score-based Models",
        
        description: "This post explores the unification of DDPM and Score-based Models in diffusion generative modeling. We show how x-prediction and score-prediction are fundamentally equivalent, and how both can be viewed through the lens of Stochastic Differential Equations (SDEs).",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/Note-Diffusion-DDPM-and-NCSN/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-i-set-up-my-personal-page",
          title: 'I set up my personal Page!',
          description: "",
          section: "News",},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%72%75%6E%7A%65%72.%74%69%61%6E@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/Gua927", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/RunzerT60347", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
