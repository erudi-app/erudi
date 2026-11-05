// Standalone webpack build (replaces @electron-forge/plugin-webpack).
//
// Produces the SAME .webpack/ layout the Electron main process expects:
//   .webpack/main/index.js                          (main process — package.json "main")
//   .webpack/renderer/main_window/index.html        (renderer document)
//   .webpack/renderer/main_window/index.js          (renderer bundle)
//   .webpack/renderer/main_window/preload.js        (contextBridge preload)
//
// Dev: `webpack serve` serves the renderer at http://localhost:3000/ ; main.js
// loads that URL when the app is not packaged. Prod: main.js loadFile()s the
// built index.html (see src/main.js entry resolution).

const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const baseRules = require("./webpack.rules");

const isDev = process.env.NODE_ENV !== "production";
const mode = isDev ? "development" : "production";

const RENDERER_OUT = path.resolve(__dirname, ".webpack/renderer/main_window");

// Renderer rules are deliberately NOT derived from webpack.rules: that set
// carries the @vercel/webpack-asset-relocator-loader, which is for native node
// modules in the MAIN process and injects `__dirname` references that crash a
// browser/web renderer ("__dirname is not defined"). The renderer only needs
// JSX, the Tailwind/PostCSS CSS pipeline, and image assets.
const rendererRules = [
  {
    test: /\.jsx?$/,
    exclude: /node_modules/,
    use: { loader: "babel-loader", options: { presets: [["@babel/preset-react", { runtime: "classic" }]] } },
  },
  {
    test: /\.css$/,
    use: [
      "style-loader",
      "css-loader",
      {
        loader: "postcss-loader",
        options: {
          postcssOptions: {
            plugins: [require("tailwindcss"), require("autoprefixer")],
          },
        },
      },
    ],
  },
  {
    test: /\.(png|jpe?g|gif|svg)$/i,
    type: "asset/resource",
  },
  {
    // Self-hosted fonts (Montserrat via @fontsource, KaTeX math fonts). Inlined
    // as data URIs so they load offline and over file:// in the packaged app,
    // with no extra fetch. KaTeX's CSS also references .ttf fallbacks, which
    // css-loader resolves too — hence the ttf extension.
    test: /\.(woff2?|ttf)$/i,
    type: "asset/inline",
  },
];

const resolve = { extensions: [".js", ".jsx", ".json"] };

/** Main process bundle. */
const main = {
  name: "main",
  mode,
  target: "electron-main",
  entry: "./src/main.js",
  output: { path: path.resolve(__dirname, ".webpack/main"), filename: "index.js" },
  module: { rules: baseRules },
  resolve,
  // Keep the real runtime __dirname so main.js can resolve sibling resources
  // (renderer index.html, preload, icons) relative to the .webpack/main dir.
  node: { __dirname: false, __filename: false },
  devtool: isDev ? "source-map" : false,
};

/** Preload — runs in the isolated context bridging renderer <-> main. */
const preload = {
  name: "preload",
  mode,
  target: "electron-preload",
  entry: "./src/preload.js",
  output: { path: RENDERER_OUT, filename: "preload.js" },
  module: { rules: baseRules },
  resolve,
  node: { __dirname: false, __filename: false },
  devtool: isDev ? "source-map" : false,
};

/** Renderer — React app. Web target since nodeIntegration is off. */
const renderer = {
  name: "renderer",
  mode,
  target: "web",
  entry: "./src/renderer.js",
  output: {
    path: RENDERER_OUT,
    filename: "index.js",
    // Relative refs so the document works when loaded via file:// in prod.
    publicPath: isDev ? "/" : "",
  },
  module: { rules: rendererRules },
  resolve,
  plugins: [
    new HtmlWebpackPlugin({
      template: "./public/index.html",
      inject: true,
      scriptLoading: "blocking",
      // CSP for the prod (file://) load, where main.js's onHeadersReceived does
      // not apply. Mirrors the policy the forge renderer config used to inject.
      // `img-src` deliberately omits `https:`, exactly like the header in
      // src/main.js: the window never needs a remote image, and allowing the
      // scheme would let an image link written by a model, or carried by a
      // knowledge-base document, reveal the user's IP to the host it points at.
      meta: {
        "Content-Security-Policy": {
          "http-equiv": "Content-Security-Policy",
          content:
            "default-src 'self'; connect-src 'self' http://127.0.0.1:* http://localhost:*; " +
            "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; " +
            "img-src 'self' data:; font-src 'self' data:;",
        },
      },
    }),
  ],
  devtool: isDev ? "source-map" : false,
  devServer: {
    port: 3000,
    hot: true,
    static: false,
    devMiddleware: { writeToDisk: (file) => /preload\.js$/.test(file) },
  },
};

module.exports = [main, preload, renderer];
