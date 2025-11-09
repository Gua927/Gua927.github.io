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
        },{id: "post-training-free-method-for-parallel-decoding-of-autoregressive-models",
        
          title: "Training-Free Method for Parallel Decoding of Autoregressive Models",
        
        description: "This blog post investigates the possibility of parallel decoding for autoregressive models. The author notes that autoregressive and diffusion models both fundamentally model data probability distributions, and that each has advantages—autoregressive models in training and diffusion models in sampling. The goal is to achieve a training-free way to perform parallel decoding with a pretrained autoregressive model, enabling low-cost accelerated generation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/Note-AR2Diff/";
          
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
