// Formal project report (max. 5 pages). Build: python scripts/build_report_pdf.py

#let title = "Brain Tumor Segmentation in MRI Using a U-Net Convolutional Neural Network"
#let author = "Emre Yoleri"
#let date = "October 2026"

#set document(title: title, author: author)
#set page(paper: "a4", margin: (x: 2.5cm, top: 2.5cm, bottom: 2.3cm),
  footer: context align(center, text(size: 9pt, counter(page).display("1"))))
#set text(font: "Times New Roman", size: 11pt, lang: "en", region: "us")
#set par(justify: true, leading: 0.62em, spacing: 0.95em, first-line-indent: 0pt)
#set heading(numbering: "1.1.")
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 2): set text(size: 11pt)
#show heading: set block(above: 1.2em, below: 0.7em)
#set math.equation(numbering: "(1)")
#set figure(gap: 0.6em)
#show figure: set block(above: 1em, below: 1em)
#show figure.caption: set text(size: 9.5pt)
#show figure.caption: it => [*#it.supplement #context it.counter.display(it.numbering).* #it.body]
#show figure.where(kind: table): set figure.caption(position: top)
#set table(stroke: (x, y) => (top: if y <= 1 { 0.6pt } else { 0pt }, bottom: 0.6pt), inset: (x: 5pt, y: 3.2pt))
#show table: set text(size: 9.5pt)
#show table.cell.where(y: 0): strong

// ---------------------------------------------------------------- title block
#align(center)[
  #text(size: 15pt, weight: "bold")[#title]
  #v(0.5em)
  #text(size: 11pt)[#author]
  #v(0.1em)
  #text(size: 10pt, style: "italic")[Technical Report · #date]
]
#v(0.6em)
#line(length: 100%, stroke: 0.5pt)
#block(inset: (x: 0.6cm))[
  #set text(size: 10pt)
  *Abstract.* Manual delineation of brain tumors in magnetic resonance imaging (MRI) is slow and varies
  between readers. This report presents a deep learning system that, for each axial MRI slice, decides
  whether a lower-grade glioma is present and segments it at pixel level. A U-Net with 7.76 M parameters
  was trained in PyTorch on the public LGG MRI Segmentation dataset (110 patients, 3,929 slices) with a
  combined binary cross-entropy and Dice loss, brain-masked intensity normalization and a strict
  patient-level data split. Post-processing parameters were tuned on the validation set only. On 17
  unseen test patients (681 slices) the model reached a global Dice coefficient of 0.822, a mean
  per-patient Dice of 0.776 and a slice-level detection accuracy of 90.0 % (sensitivity 94.6 %,
  specificity 87.5 %), after only eight training epochs. Small lesions under 500 pixels remain the main
  source of error.

  *Keywords:* brain tumor segmentation, lower-grade glioma, MRI, U-Net, Dice loss, deep learning.
]
#line(length: 100%, stroke: 0.5pt)

// ---------------------------------------------------------------- body
= Introduction

Lower-grade gliomas (LGG) are primary brain tumors that appear as a hyperintense region on the
fluid-attenuated inversion recovery (FLAIR) MRI sequence. Their extent is needed for treatment planning
and follow-up, and the shape of the lesion has been linked to its genomic subtype @buda2019. In practice,
however, the tumor is outlined by hand slice by slice, which is time-consuming and subject to
inter-reader variability @menze2015. Automatic segmentation can make this measurement fast and
reproducible.

Convolutional encoder–decoder networks, and the U-Net @ronneberger2015 in particular, are the standard
approach for biomedical image segmentation and remain highly competitive when well configured
@isensee2021. Brain tumor segmentation is nevertheless difficult because of strong class imbalance: most
slices contain no tumor, and inside tumor slices the lesion covers only a few percent of the pixels.

The objectives of this work are to (i) analyse and clean a public brain MRI dataset and split it without
patient leakage, (ii) train a U-Net that segments the FLAIR abnormality on each slice, and (iii) evaluate
it at pixel, slice and patient level on held-out patients, with an analysis of its failure cases.

= Methodology

== Dataset

The LGG MRI Segmentation dataset @buda2019 contains brain MRI from 110 patients of The Cancer Genome
Atlas lower-grade glioma collection, obtained from The Cancer Imaging Archive @clark2013. It provides
3,929 axial slices of $256 times 256$ pixels with three channels (pre-contrast, FLAIR and post-contrast)
and a manual binary mask of the FLAIR abnormality. Only 1,373 slices (34.9 %) contain tumor, and tumor
pixels make up 1.03 % of all pixels; the median tumor area is 1,529 pixels, ranging from 314 (10th
percentile) to 4,152 (90th percentile). For some patients a missing contrast sequence was
replaced by the FLAIR image in the original data.

Duplicate copies of the archive were removed and every slice was checked for a readable image and
matching mask, correct size, non-blank content and binary mask values; all 3,929 slices passed. The
data were split *by patient*, stratified on each patient's share of tumor slices (seed 42), so that no
patient contributes slices to more than one set (@tab-split). A random slice-level split would place
nearly identical neighbouring slices in the training and test sets and inflate the scores.

#figure(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center),
    table.header([Split], [Patients], [Slices], [Tumor slices]),
    [Training], [76], [2,584], [898 (34.8 %)],
    [Validation], [17], [664], [235 (35.4 %)],
    [Test], [17], [681], [240 (35.2 %)],
    table.hline(stroke: 0.4pt),
    [Total], [110], [3,929], [1,373 (34.9 %)],
  ),
  caption: [Patient-level data split.],
) <tab-split>

== Preprocessing and Augmentation

MRI intensities are not calibrated across scanners, so each channel of each image was normalized by a
z-score computed over brain (non-background) pixels only. During training, the image and mask were
augmented together with horizontal flips, rotations of up to $plus.minus 15 degree$, scaling between 0.9
and 1.1, shifts of up to 6 %, and brightness, contrast and gamma jitter.

== Model and Training

The network is a U-Net @ronneberger2015 with four down-sampling and four up-sampling levels (32 to 512
feature channels), batch normalization @ioffe2015, dropout of 0.1 in the deepest blocks and a
single-channel sigmoid output, giving 7.76 M parameters. To counter class imbalance, the loss combines
binary cross-entropy with the soft Dice loss @milletari2016:

$ cal(L) = "BCE"(p, g) + 1 - (2 sum_i p_i g_i + epsilon) / (sum_i p_i + sum_i g_i + epsilon) , $ <eq-loss>

where $p_i$ is the predicted probability and $g_i$ the ground-truth label of pixel $i$. In addition,
tumor slices were oversampled by a factor of two. The model was trained in PyTorch @paszke2019 with
AdamW @loshchilov2019 (weight decay $10^(-4)$), a one-cycle learning-rate schedule @smith2019 and a batch
size of 16 on an Apple M-series GPU. Because of a strict time budget, training ran for 5 epochs at a
peak learning rate of $10^(-3)$, followed by 3 fine-tuning epochs at $3 dot 10^(-4)$ (about 18 minutes
in total). The checkpoint with the best validation Dice (0.783, epoch 8) was kept.

== Post-processing and Evaluation

At inference, the prediction is averaged with the prediction on the horizontally flipped slice. The
probability threshold and a minimum connected-component area were then selected by a grid search on
the *validation* set (threshold 0.3–0.7, minimum area 0–100 pixels, criterion: mean per-slice Dice);
the selected values, threshold 0.7 and minimum area 100 pixels, were applied unchanged to the test set.

Segmentation quality is measured with the Dice coefficient @dice1945, $"Dice" = 2|P inter G| \/ (|P| +
|G|)$, and the intersection over union (IoU). Both are reported globally (all test pixels pooled), as a
mean over tumor slices, as a mean over all slices (an empty prediction on an empty slice counts as 1),
and per patient on the reconstructed 3D volume. Slice-level detection is evaluated by treating a
non-empty predicted mask as a positive "tumor present" decision.

= Results

== Quantitative Results

@tab-results summarizes the test results on the 17 unseen patients. The U-Net alone reaches a global Dice
of 0.814. Post-processing raises it to 0.822 and mainly improves slice-level detection: removing small
false-positive regions on healthy slices increases specificity from 78.0 % to 87.5 % (false positives
fall from 97 to 55 slices), at a cost of only 0.8 points of sensitivity (two additional missed slices).
The mean Dice over all slices rises from 0.779 to 0.840 for the same reason.

#figure(
  table(
    columns: (2.4fr, 1fr, 1fr),
    align: (left, center, center),
    table.header([Metric], [U-Net (thr. 0.5)], [\+ Post-processing]),
    [Dice, global], [0.814], [*0.822*],
    [IoU, global], [0.686], [*0.698*],
    [Mean Dice, tumor slices], [*0.778*], [0.774],
    [Mean Dice, all slices], [0.779], [*0.840*],
    [Mean Dice per patient (3D)], [0.768], [*0.776*],
    [Pixel precision / recall], [0.791 / *0.839*], [*0.833* / 0.811],
    table.hline(stroke: 0.4pt),
    [Detection accuracy], [0.841], [*0.900*],
    [Detection sensitivity], [*0.954*], [0.946],
    [Detection specificity], [0.780], [*0.875*],
  ),
  caption: [Test results on 17 unseen patients (681 slices, 240 with tumor). Best value in bold.],
) <tab-results>

== Training Behaviour

@fig-train shows the training curves. The training loss decreases steadily, while the validation Dice
rises from 0.675 to 0.783 but is still noisy (it drops to 0.571 at epoch 7) and still increasing at the
last epoch. The model has therefore not converged, and the reported results are a lower bound on what
this architecture can reach.

#figure(
  image("/results/figures/training_curves.png", width: 92%),
  caption: [Training and validation loss (left) and validation Dice and IoU (right) per epoch.],
) <fig-train>

== Qualitative Results and Error Analysis

The error analysis in @tab-size and @fig-dice shows that tumor size is the dominant factor. Slices with lesions under 500
pixels, typically the first and last slices of a tumor, reach a mean Dice of only 0.50, while medium-size
lesions reach 0.89. Part of the small-lesion error is caused by the 100-pixel minimum-area filter,
which also removes very small true lesions. At patient level, predicted tumor volumes agree well with
the ground truth (median per-patient Dice 0.822); the worst patient (Dice 0.40) has a small total tumor
volume that the model over-estimates by a factor of about two.

#grid(
  columns: (1fr, 2fr),
  gutter: 0.5cm,
  align: horizon,
  [#figure(
    table(
      columns: (1.3fr, 0.8fr, 0.9fr),
      align: (left, center, center),
      table.header([Area (px)], [Slices], [Dice]),
      [< 500], [45], [0.50],
      [500–1,500], [81], [0.78],
      [1,500–4,000], [101], [*0.89*],
      [> 4,000], [13], [0.79],
    ),
    caption: [Mean test Dice by tumor size.],
  ) <tab-size>],
  [#figure(
    image("/results/figures/test_dice_analysis.png", width: 100%),
    caption: [Dice per tumor slice, Dice vs. tumor size, and predicted vs. true tumor volume per patient.],
  ) <fig-dice>],
)

Typical predictions are shown in @fig-pred. Large, well-contrasted tumors are segmented accurately (Dice 0.96);
most remaining errors lie on the faint tumor boundary, where the FLAIR hyperintensity fades gradually.
The bottom row shows a characteristic failure: a very small lesion is missed while a small false
positive appears elsewhere.

#figure(
  placement: auto,
  image("/report/figures/test_predictions_compact.png", width: 74%),
  caption: [Test predictions. Columns: input slice, ground truth, predicted probability, error map
  (green: true positive, yellow: false negative, red: false positive).],
) <fig-pred>

= Discussion

The results show that a standard U-Net, trained with a Dice-based loss and a careful patient-level
protocol, segments lower-grade gliomas well even with a very short training budget. The combined BCE
and Dice loss together with tumor-slice oversampling was sufficient to handle the strong class
imbalance: pixel recall on the raw model is high (0.839), and the main residual problem is false
positives on healthy slices rather than missed tumors. Tuning the threshold and minimum region area on
the validation set addresses this cheaply and transparently, and because the parameters were never
selected on test data, the test scores are an unbiased estimate.

For comparison, Buda et al. @buda2019 report a mean per-patient Dice of about 0.82 on the same dataset
with a longer-trained U-Net. The present model reaches 0.776 after eight epochs, and the still rising
validation curve suggests that most of this gap is due to under-training rather than to the method.

The main limitations are the short training (the model has not converged), the small test set of 17
patients, which makes per-patient estimates noisy, and the 2D slice-wise design, which ignores context
from neighbouring slices and contributes to errors on small lesions at the ends of a tumor. The
annotations themselves are uncertain at the faint tumor boundary. The system is an educational
prototype and has not been validated for clinical use.

= Conclusion

A U-Net for brain tumor segmentation was developed in PyTorch and evaluated on the LGG MRI Segmentation
dataset with a strict patient-level split. On 17 unseen patients it reached a global Dice of 0.822, a
mean per-patient Dice of 0.776 and a tumor detection accuracy of 90.0 %, with validation-tuned
post-processing that raised specificity from 78.0 % to 87.5 %. The model is also available in an
interactive Streamlit web application. Future work includes training to convergence, using a pretrained
encoder and neighbouring slices as additional input channels, size-aware post-processing to recover
small lesions, and patient-level cross-validation.

#v(0.3em)
#{
  set text(size: 9.5pt)
  set par(leading: 0.5em, spacing: 0.55em)
  show heading: set block(above: 1.2em, below: 0.7em)
  bibliography("references.yml", title: [References], style: "ieee")
}
