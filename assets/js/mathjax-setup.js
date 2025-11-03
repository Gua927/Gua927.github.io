window.MathJax = {
  tex: {
    tags: "ams",
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    // Enable automatic line breaking for long equations
    processEscapes: true,
    processEnvironments: true,
    processRefs: true,
    digits: /^(?:[0-9]+(?:\{,\}[0-9]{3})*(?:\.[0-9]*)?|\.[0-9]+)/,
    // Improve parsing for complex inline math
    maxBuffer: 5 * 1024,
  },
  chtml: {
    // Enable responsive scaling
    scale: 1,
    minScale: 0.5,
    matchFontHeight: true,
    // Improve display for long equations
    linebreaks: {
      automatic: false, // Set to false to prevent unexpected breaks
      width: "container",
    },
  },
  options: {
    renderActions: {
      addCss: [
        200,
        function (doc) {
          const style = document.createElement("style");
          style.innerHTML = `
          .mjx-container {
            color: inherit;
          }
          /* Ensure long equations can scroll horizontally */
          mjx-container[display="true"] {
            overflow-x: auto;
            overflow-y: hidden;
            max-width: 100%;
          }
          /* Better scrollbar for equations */
          mjx-container[display="true"]::-webkit-scrollbar {
            height: 6px;
          }
          mjx-container[display="true"]::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.05);
            border-radius: 3px;
          }
          mjx-container[display="true"]::-webkit-scrollbar-thumb {
            background: rgba(0,0,0,0.2);
            border-radius: 3px;
          }
          mjx-container[display="true"]::-webkit-scrollbar-thumb:hover {
            background: rgba(0,0,0,0.3);
          }
        `;
          document.head.appendChild(style);
        },
        "",
      ],
    },
  },
};
