const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Spencer Hodge";
pres.title = "Task Embeddings and Occupational Mobility";

// Design constants
const TITLE_FONT = "Calibri";
const BODY_FONT = "Calibri";
const CLR_DARK = "2D2D2D";
const CLR_SECONDARY = "777777";
const CLR_ACCENT = "4A7FB5";
const CLR_SLIDENUM = "999999";

// (Placeholder function removed — using addImage with actual figures)

function addSlideNumber(slide, num) {
  slide.addText(String(num), {
    x: 9.0,
    y: 5.2,
    w: 0.7,
    h: 0.3,
    fontSize: 10,
    color: CLR_SLIDENUM,
    align: "right",
    fontFace: BODY_FONT,
  });
}

function addTopRule(slide) {
  slide.addShape(pres.shapes.LINE, {
    x: 0.5,
    y: 0.15,
    w: 9.0,
    h: 0,
    line: { color: CLR_ACCENT, width: 1.5 },
  });
}

function addSlideTitle(slide, title) {
  slide.addText(title, {
    x: 0.5,
    y: 0.3,
    w: 9.0,
    h: 0.6,
    fontSize: 28,
    fontFace: TITLE_FONT,
    bold: true,
    color: CLR_DARK,
    margin: 0,
  });
}

// ─── SLIDE 1: Title ───
const slide1 = pres.addSlide();
slide1.background = { color: "FFFFFF" };

slide1.addText("Task Embeddings and\nOccupational Mobility", {
  x: 0.5,
  y: 1.2,
  w: 9.0,
  h: 1.4,
  fontSize: 32,
  fontFace: TITLE_FONT,
  bold: true,
  color: CLR_DARK,
  align: "center",
  lineSpacingMultiple: 1.15,
});

slide1.addText("Spencer Hodge", {
  x: 0.5,
  y: 2.7,
  w: 9.0,
  h: 0.5,
  fontSize: 18,
  fontFace: BODY_FONT,
  color: CLR_DARK,
  align: "center",
});

slide1.addText([
  { text: "Bagwell Center for the Study of Markets and Economic Opportunity", options: { breakLine: true } },
  { text: "Kennesaw State University", options: { breakLine: true } },
  { text: "Undergraduate Research Fellowship, 2025\u201326" },
], {
  x: 0.5,
  y: 3.3,
  w: 9.0,
  h: 1.2,
  fontSize: 14,
  fontFace: BODY_FONT,
  color: CLR_SECONDARY,
  align: "center",
  lineSpacingMultiple: 1.4,
});

slide1.addShape(pres.shapes.LINE, {
  x: 3.5,
  y: 3.2,
  w: 3.0,
  h: 0,
  line: { color: CLR_ACCENT, width: 1 },
});

// ─── SLIDE 2: Motivation ───
const slide2 = pres.addSlide();
slide2.background = { color: "FFFFFF" };
addTopRule(slide2);
addSlideTitle(slide2, "Motivation");
addSlideNumber(slide2, 2);

slide2.addText([
  { text: "New technology is reshaping the labor market.", options: { breakLine: true, bold: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "If your job is disrupted, what can you switch to?", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "We propose a way to answer that question \u2014 and a test for whether it works.", options: {} },
], {
  x: 0.5,
  y: 1.1,
  w: 3.0,
  h: 4.0,
  fontSize: 14,
  fontFace: BODY_FONT,
  color: CLR_DARK,
  lineSpacingMultiple: 1.3,
  valign: "top",
});

slide2.addImage({ path: "figures/fig1_ai_exposure.png", x: 3.8, y: 0.9, w: 5.4, h: 4.0 });

// ─── SLIDE 3: What is a Sentence Embedding? (NEW) ───
const slideEmbed = pres.addSlide();
slideEmbed.background = { color: "FFFFFF" };
addTopRule(slideEmbed);
addSlideTitle(slideEmbed, "What is a Sentence Embedding?");
addSlideNumber(slideEmbed, 3);

slideEmbed.addImage({ path: "figures/fig8_embedding_flowchart.png", x: 0.5, y: 1.0, w: 3.8, h: 4.0 });
slideEmbed.addImage({ path: "figures/fig9_word_analogy.png", x: 4.6, y: 1.0, w: 3.9, h: 4.0 });

// ─── SLIDE 4: Tasks in Semantic Space ───
const slide3 = pres.addSlide();
slide3.background = { color: "FFFFFF" };
addTopRule(slide3);
addSlideTitle(slide3, "Tasks in Semantic Space");
addSlideNumber(slide3, 4);

slide3.addText([
  { text: "1.  Embed 2,087 O*NET task descriptions into a continuous space (MPNet, 768d).", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "2.  Represent each occupation as a distribution over embedded tasks.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "3.  Test against 89,329 observed job switches (CPS, 2015\u20132024)." },
], {
  x: 0.5,
  y: 1.1,
  w: 2.8,
  h: 4.0,
  fontSize: 14,
  fontFace: BODY_FONT,
  color: CLR_DARK,
  lineSpacingMultiple: 1.3,
  valign: "top",
});

slide3.addImage({ path: "figures/fig3_task_scatter.png", x: 3.5, y: 1.0, w: 5.5, h: 4.0 });

// ─── SLIDE 5: Example: Why Embeddings Matter (NEW) ───
const slideExample = pres.addSlide();
slideExample.background = { color: "FFFFFF" };
addTopRule(slideExample);
addSlideTitle(slideExample, "Example: Why Embeddings Matter");
addSlideNumber(slideExample, 5);

slideExample.addImage({ path: "figures/fig5_shared_dwas.png", x: 0.5, y: 1.0, w: 3.9, h: 4.0 });
slideExample.addImage({ path: "figures/fig6_embedding_similarity.png", x: 4.7, y: 1.0, w: 4.8, h: 4.0 });

// ─── SLIDE 6: What Drives the Improvement? ───
const slide5 = pres.addSlide();
slide5.background = { color: "FFFFFF" };
addTopRule(slide5);
addSlideTitle(slide5, "What Drives the Improvement?");
addSlideNumber(slide5, 6);

slide5.addText("Crossing two choices to isolate what drives predictive gain:", {
  x: 0.5,
  y: 1.1,
  w: 9.0,
  h: 0.4,
  fontSize: 14,
  fontFace: BODY_FONT,
  color: CLR_DARK,
});

const headerOpts = { fill: { color: CLR_ACCENT }, color: "FFFFFF", bold: true, align: "center", valign: "middle", fontSize: 13, fontFace: BODY_FONT };
const cellOpts = { fill: { color: "F8F8F8" }, color: CLR_DARK, align: "center", valign: "middle", fontSize: 13, fontFace: BODY_FONT };
const starOpts = { fill: { color: "E8F0F8" }, color: CLR_DARK, align: "center", valign: "middle", fontSize: 13, fontFace: BODY_FONT, bold: true };

const tableData = [
  [
    { text: "", options: { ...headerOpts } },
    { text: "Simple average", options: { ...headerOpts } },
    { text: "Optimal transport", options: { ...headerOpts } },
  ],
  [
    { text: "O*NET ratings", options: { ...headerOpts } },
    { text: "Baseline", options: { ...cellOpts } },
    { text: "Identity control", options: { ...cellOpts } },
  ],
  [
    { text: "Text embeddings", options: { ...headerOpts } },
    { text: "Centroid (primary)", options: { ...starOpts } },
    { text: "Wasserstein", options: { ...starOpts } },
  ],
];

slide5.addTable(tableData, {
  x: 1.0,
  y: 1.8,
  w: 8.0,
  h: 1.8,
  colW: [2.2, 2.9, 2.9],
  border: { pt: 0.5, color: "DDDDDD" },
  rowH: [0.5, 0.5, 0.5],
});

slide5.addText("Validation: 89,329 verified transitions from the CPS. Primary metric: McFadden\u2019s pseudo-R\u00b2.", {
  x: 0.5,
  y: 4.0,
  w: 9.0,
  h: 0.5,
  fontSize: 14,
  fontFace: BODY_FONT,
  color: CLR_DARK,
});

// ─── SLIDE 7: Main Result ───
const slide6 = pres.addSlide();
slide6.background = { color: "FFFFFF" };
addTopRule(slide6);
addSlideTitle(slide6, "Main Result: Embeddings Dominate");
addSlideNumber(slide6, 7);

slide6.addImage({ path: "figures/fig2_pseudo_r2.png", x: 0.5, y: 1.0, w: 9.0, h: 4.1 });

// ─── SLIDE 8: Scope ───
const slide7 = pres.addSlide();
slide7.background = { color: "FFFFFF" };
addTopRule(slide7);
addSlideTitle(slide7, "What the Measure Captures");
addSlideNumber(slide7, 8);

slide7.addText([
  { text: "Supply-side feasibility, not demand.", options: { breakLine: true, bold: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Realized destinations ranked highly (MPR = 0.74).", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Aggregate inflows are demand-dominated (\u03c1 = 0.80 with BLS openings).", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Orthogonal to AI exposure (r = \u22120.02) and routine task intensity (r = \u22120.06). Distinct constructs.", options: {} },
], {
  x: 0.5,
  y: 1.1,
  w: 3.5,
  h: 4.0,
  fontSize: 14,
  fontFace: BODY_FONT,
  color: CLR_DARK,
  lineSpacingMultiple: 1.3,
  valign: "top",
});

slide7.addImage({ path: "figures/fig4_scope.png", x: 4.5, y: 1.0, w: 4.9, h: 4.0 });

// ─── SLIDE 9: Limitations ───
const slide8 = pres.addSlide();
slide8.background = { color: "FFFFFF" };
addTopRule(slide8);
addSlideTitle(slide8, "Limitations and Scope");
addSlideNumber(slide8, 9);

slide8.addText([
  { text: "Completed transitions only \u2014 workers blocked by licensing are absent.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Switching cost identification failed: occupation-mean wages \u2260 entry wages.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Single embedding model (MPNet). Domain-specific models could differ.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Cross-sectional \u2014 task content treated as static.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Reallocation forecasts need vacancy, wage, geographic, and institutional data." },
], {
  x: 0.5,
  y: 1.1,
  w: 9.0,
  h: 4.0,
  fontSize: 14,
  fontFace: BODY_FONT,
  color: CLR_DARK,
  lineSpacingMultiple: 1.3,
  valign: "top",
});

// ─── SLIDE 10: Takeaways ───
const slide9 = pres.addSlide();
slide9.background = { color: "FFFFFF" };
addTopRule(slide9);
addSlideTitle(slide9, "Takeaways");
addSlideNumber(slide9, 10);

slide9.addText([
  { text: "Embeddings improve transition prediction by 74.9% over O*NET baselines (pseudo-R\u00b2: 14.1% vs. 6\u20138%).", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "The gain is from the embedding representation, not optimal transport.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Captures within-origin pathway feasibility, not aggregate allocation.", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Orthogonal to AI exposure (r = \u22120.02) and routine task intensity (r = \u22120.06).", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "Stable pre/post COVID (\u0394\u03b1 < 1%, p = 0.72)." },
], {
  x: 0.5,
  y: 1.1,
  w: 9.0,
  h: 4.2,
  fontSize: 15,
  fontFace: BODY_FONT,
  color: CLR_DARK,
  lineSpacingMultiple: 1.3,
  valign: "top",
});

// ─── SLIDE 11: Thank You ───
const slide10 = pres.addSlide();
slide10.background = { color: "FFFFFF" };

slide10.addText("Thank You", {
  x: 0.5,
  y: 1.8,
  w: 9.0,
  h: 0.8,
  fontSize: 32,
  fontFace: TITLE_FONT,
  bold: true,
  color: CLR_DARK,
  align: "center",
});

slide10.addText([
  { text: "Spencer Hodge", options: { breakLine: true } },
  { text: "shodge33@students.kennesaw.edu" },
], {
  x: 0.5,
  y: 3.0,
  w: 9.0,
  h: 0.8,
  fontSize: 16,
  fontFace: BODY_FONT,
  color: CLR_DARK,
  align: "center",
  lineSpacingMultiple: 1.5,
});

slide10.addText("Code & data: github.com/spencer2718/task-space-model", {
  x: 0.5,
  y: 4.0,
  w: 9.0,
  h: 0.5,
  fontSize: 14,
  fontFace: BODY_FONT,
  color: CLR_ACCENT,
  align: "center",
});

slide10.addShape(pres.shapes.LINE, {
  x: 3.5,
  y: 2.85,
  w: 3.0,
  h: 0,
  line: { color: CLR_ACCENT, width: 1 },
});

// ─── BACKUP SLIDE B1: Data Pipeline ───
const slideBackup = pres.addSlide();
slideBackup.background = { color: "FFFFFF" };
addTopRule(slideBackup);
addSlideTitle(slideBackup, "Data Pipeline");
addSlideNumber(slideBackup, "B1");

slideBackup.addImage({ path: "figures/fig7_sankey_pipeline.png", x: 0.5, y: 1.0, w: 9.0, h: 4.2 });

// ─── Write file ───
pres.writeFile({ fileName: "task_embeddings_deck.pptx" })
  .then(() => console.log("Done: task_embeddings_deck.pptx"))
  .catch(err => console.error(err));