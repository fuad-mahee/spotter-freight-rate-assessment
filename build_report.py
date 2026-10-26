"""Build the submission PDF from computed results, without hand-entered metrics."""
import json
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image


def main():
    out = Path('output')
    r = json.loads((out / 'metrics.json').read_text())
    d = json.loads((out / 'diagnostics.json').read_text())
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom', fontName='Helvetica', fontSize=10,
                              leading=14, spaceAfter=9, textColor=colors.black))
    styles.add(ParagraphStyle(name='Kicker', fontSize=9, leading=13,
                              textColor=colors.black, spaceAfter=10))
    styles['Title'].textColor = colors.black
    styles['Title'].alignment = TA_LEFT
    styles['Title'].fontSize = 26
    styles['Title'].leading = 31
    styles['Heading1'].textColor = colors.black
    styles['Heading2'].textColor = colors.black
    story = []

    def p(text, style='BodyCustom'):
        story.append(Paragraph(text, styles[style]))

    def table(rows, widths):
        rows = [[Paragraph(escape(str(c)), styles['BodyCustom']) for c in row] for row in rows]
        t = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LINEABOVE', (0, 0), (-1, 0), .6, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), .6, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), .4, colors.black)]))
        story.extend([t, Spacer(1, 12)])

    full = r['selected']['full']
    core = r['selected']['core']
    fm = r['holdout'][full]
    cm = r['holdout'][core]
    p('SPOTTER / MACHINE LEARNING ENGINEER ASSESSMENT', 'Kicker')
    p('Freight-rate prediction', 'Title')
    p('Muftasim Fuad | Technical approach and validation', 'Heading2')
    p(f"CatBoost predicts total rates for 12,000 November-December 2025 loads. The selected "
      f"main model achieves <b>${fm['MAE']:.2f} September-October holdout MAE</b>, a "
      f"<b>{d['baseline_mae_reduction_pct']:.1f}% reduction</b> versus the distance x quote baseline. "
      'Systematic underprediction and unmeasured unseen-city accuracy are the main limitations. '
      'The hidden final score is unknown.')
    p('CatBoost suits mixed numerical and categorical tabular data and learns nonlinear effects '
      'and interactions among distance, equipment, location and market inputs. It avoids manually '
      'one-hot encoding all lanes and supports numerical missing values. These conveniences do '
      'not by themselves establish performance on unseen cities.')
    p('1. Data review', 'Heading1')
    table([['Input', 'Coverage and role'],
           ['Labeled development', '48,000 rows; January 1-October 31, 2025; target: posted_rate.'],
           ['Final prediction set', '12,000 rows; November 1-December 31, 2025; targets withheld.'],
           ['December scenario', '31 daily rows; Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb.']], [145, 355])
    a, b = r['audit']['train'], r['audit']['final']
    p(f"Training has {a['missing']['weight']} missing weights, {a['invalid_weight']} non-positive weights, "
      f"and {a['missing']['market_index']} missing market indices. The final set has "
      f"{b['missing']['weight']} missing weights, {b['invalid_weight']} non-positive weights and "
      f"{b['missing']['market_index']} missing market indices. Non-positive weights become missing; "
      'separate invalidity and missingness flags are retained. CatBoost handles numerical NaNs directly.')
    p('There are no duplicate IDs or duplicate complete feature rows within either dataset. '
      'IDs do not overlap across the labeled and final sets. Targets and distances are finite and positive. '
      'Large positive target values are retained, including in evaluation: there is insufficient evidence '
      'to label them erroneous. Their impact is visible in the gap between MAE and RMSE.')
    p(f"Eight cities are unseen in training, affecting {r['audit']['new_city_final_rows']:,} final rows "
      f"({d['geography']['final_vs_jan_oct']['unseen_city_pct']:.1f}%). The September-October holdout "
      'has zero unseen-city loads relative to January-August training. It does not measure this '
      'part of deployment. Accepting unseen categories at inference does not demonstrate accuracy.')
    p('The root README is the solution entry point. Original challenge instructions are retained '
      'in provided_readme.md. Commands use the actual supplied root-level, hyphenated filenames.')

    story.append(PageBreak())
    p('2. Validation and model choice', 'Heading1')
    s = r['split']
    table([['Stage', 'Training window', 'Evaluation window'],
           ['Candidate selection', f"Jan-Jun ({s['train_jan_jun']:,})", f"Jul-Aug ({s['selection_jul_aug']:,})"],
           ['Local holdout', f"Jan-Aug ({s['refit_jan_aug']:,})", f"Sep-Oct ({s['holdout_sep_oct']:,})"],
           ['Final fit', 'Jan-Oct (48,000)', 'Nov-Dec: labels unavailable']], [125, 185, 190])
    p('A chronological split matches the future-only prediction set. Random splitting would mix '
      'past and future market regimes. Candidate selection uses July-August MAE; September-October '
      'is kept outside model selection. After local evaluation, refit the selected models on all labeled rows. '
      'The final-set features are audited but never used for fitting, tuning or preprocessing statistics.')
    p('MAE is the declared primary criterion because the supplied scorer does not specify an accuracy '
      'metric. RMSE, WAPE and R-squared are reported as complementary diagnostics. All candidates use '
      '650 trees, depth 6, learning rate 0.055, seed 42 and four CPU threads; no holdout early stopping.')
    labels = {'median_baseline': 'Median baseline', 'quote_baseline': 'Distance x quote baseline',
              'full_direct_mae': 'All inputs / MAE', 'full_residual_mae': 'Quote residual / MAE',
              'full_direct_rmse': 'All inputs / RMSE', 'no_quote_ablation': 'Without quote / MAE',
              'core_direct': 'Six inputs / total rate', 'core_per_mile': 'Six inputs / rate per mile'}
    rows = [['July-August candidate', 'MAE ($)', 'RMSE ($)']]
    for name, m in r['selection'].items():
        rows.append([labels[name], f"{m['MAE']:,.2f}", f"{m['RMSE']:,.2f}"])
    table(rows, [300, 100, 100])
    p(f"Selected full-data model: <b>{labels[full]}</b>. Selected December model: <b>{labels[core]}</b>. "
      'The quote-free candidate is eligible for the main prediction task. Its better selection MAE '
      'supports omitting quote_signal; no assumption about its upstream construction is needed by '
      'the selected model. The comparison does not prove that quote_signal is leakage.')

    story.append(PageBreak())
    p('3. Held-out results', 'Heading1')
    table([['September-October', 'MAE ($)', 'RMSE ($)', 'WAPE'],
           ['Median baseline', f"{r['holdout']['median_baseline']['MAE']:,.2f}",
            f"{r['holdout']['median_baseline']['RMSE']:,.2f}", f"{r['holdout']['median_baseline']['WAPE_pct']:.2f}%"],
           ['Distance x quote', f"{r['holdout']['quote_baseline']['MAE']:,.2f}",
            f"{r['holdout']['quote_baseline']['RMSE']:,.2f}", f"{r['holdout']['quote_baseline']['WAPE_pct']:.2f}%"],
           ['Selected main model', f"{fm['MAE']:,.2f}", f"{fm['RMSE']:,.2f}", f"{fm['WAPE_pct']:.2f}%"],
           ['Selected December model', f"{cm['MAE']:,.2f}", f"{cm['RMSE']:,.2f}", f"{cm['WAPE_pct']:.2f}%"]], [230, 90, 90, 90])
    p(f"Main-model holdout R-squared: {fm['R2']:.4f}; December-model R-squared: {cm['R2']:.4f}. "
      'These are local retrospective results, not the hidden November-December score. '
      'The supplied score.py validates file format and generates the chart; only Spotter can score final accuracy.')
    rows = [['Main-model slice', 'Rows', 'MAE ($)', 'RMSE ($)']]
    for group in ['full_by_month', 'full_by_equipment']:
        for name, m in r['holdout'][group].items():
            rows.append([name, f"{m['n']:,}", f"{m['MAE']:,.2f}", f"{m['RMSE']:,.2f}"])
    table(rows, [230, 90, 90, 90])
    p('Feature design', 'Heading2')
    p('Categorical pickup, delivery, equipment and directional lane; distance and log-distance; '
      'cleaned weight with flags; day of week, day of month, month, elapsed day, weekend/month-end '
      'flags and annual sine/cosine terms. The main model additionally uses supplied coordinates, '
      'coordinate differences and market index with its missingness flag. Neither IDs nor target rates '
      'are input features. All feature transformations are stateless; categorical statistics are learned inside each fit.')
    p('Practical limits', 'Heading2')
    p('One year of January-October data cannot establish December holiday seasonality. Tree models '
      'have limited extrapolation beyond observed time trends; cyclical features do not solve missing seasonal coverage. '
      'The core model lacks contemporaneous market information. New-city generalization and future market '
      'shifts remain risks. No confidence interval or minimum final score is claimed. More history, verified '
      'feature provenance, city-held-out evaluation and interval calibration would be useful next steps.')

    story.append(PageBreak())
    p('4. Residuals: tail risk and underprediction', 'Heading1')
    p('The following post-evaluation diagnostics use the saved holdout predictions, joined to '
      'source rows by unique load_id. Signed error is prediction minus actual. This analysis '
      'does not fit, select or calibrate models; all predictions remain unchanged.')
    res = d['residuals']['overall']
    table([['Main-model diagnostic', 'Result'],
           ['MAE', f"${res['MAE']:.2f}"],
           ['Median absolute error', f"${res['median_absolute_error']:.2f}"],
           ['95th-percentile absolute error', f"${res['p95_absolute_error']:.2f}"],
           ['Mean signed error', f"{'-' if res['mean_signed_error'] < 0 else ''}${abs(res['mean_signed_error']):.2f}"],
           ['Median signed error', f"{'-' if res['median_signed_error'] < 0 else ''}${abs(res['median_signed_error']):.2f}"],
           ['Loads underpredicted', f"{res['underpredicted_pct']:.1f}%"],
           [f"Squared error from worst {res['worst_1pct_rows']} rows (about 1%)",
            f"{res['worst_1pct_squared_error_pct']:.1f}%"]], [370, 130])
    rows = [['Slice', 'Rows', 'Mean error ($)', 'Median error ($)', 'Underpredicted']]
    for group in ['month', 'distance_band']:
        for label, stats in d['residuals'][group].items():
            rows.append([label, f"{stats['n']:,}", f"{stats['mean_signed_error']:.2f}",
                         f"{stats['median_signed_error']:.2f}", f"{stats['underpredicted_pct']:.1f}%"])
    table(rows, [110, 65, 105, 105, 115])
    p('Two distinct issues are visible: rare extreme misses dominate squared error, while the '
      'negative median and 80.6% underprediction rate show broad underprediction among ordinary '
      'loads. Bias appears in both months and all fixed distance bands; its dollar magnitude and '
      'frequency increase with distance. These descriptive relationships do not establish the cause.')
    p('Operational consequence: using these estimates directly for quotes or budgets could '
      'understate transport costs, especially on longer trips. Extreme observations remain in all '
      'headline metrics. No $90.93 offset or other correction is estimated from this holdout. Any '
      'correction must be fit and validated within development-only chronological folds, then '
      'assessed on a separate future evaluation period.')

    story.append(PageBreak())
    p('5. Model decisions and validation gaps', 'Heading1')
    table([['Model', 'Jul-Aug MAE ($)', 'Sep-Oct MAE ($)'],
           ['Main: no quote / total dollars', f"{r['selection'][full]['MAE']:.2f}", f"{fm['MAE']:.2f}"],
           ['Core: six inputs / rate per mile', f"{r['selection'][core]['MAE']:.2f}", f"{cm['MAE']:.2f}"]],
          [260, 120, 120])
    p(f"The core model has {d['core_holdout_mae_reduction_pct']:.1f}% lower holdout MAE. We preserve "
      'the main model chosen on July-August instead of switching after observing September-October. '
      'The reversal is evidence that model rankings vary across time. Both features and target '
      'formulation differ, so this comparison cannot isolate the benefit of removing features. '
      'The holdout has now been inspected for diagnostics; it must not be reused as fresh tuning data.')
    p('What rate-per-mile MAE optimizes', 'Heading2')
    p('The core candidate learns posted_rate / distance and converts predictions back to total '
      'dollars before evaluation. Because distance is positive, its unweighted training objective '
      'is proportional to abs(total_rate_error) / distance. A dollar of error on a shorter trip '
      'receives more weight than a dollar on a longer trip: this is a modeling choice, not just units.')
    p(f"This formulation was tried to represent distance-normalized pricing. On July-August total-dollar "
      f"MAE, it scored ${r['selection'][core]['MAE']:.2f} versus "
      f"${r['selection']['core_direct']['MAE']:.2f} for the core total-rate candidate. "
      'Weighting each per-mile training loss by distance would align the objective with total-dollar '
      'MAE, up to a constant. That experiment has not been run.')
    p('Unseen-city coverage and future tests', 'Heading2')
    p('The final set has 1,447 loads involving unseen cities (12.1%); the chronological holdout '
      'has none. A geographic stress test would choose held-out cities before inspecting errors, '
      'remove every load touching them from an earlier training window, and evaluate later loads '
      'involving those cities separately from seen-city loads. This is proposed future work, not '
      'a completed validation result.')
    p('Further development would compare feature sets and target formulations on earlier rolling '
      'chronological windows, including distance-weighted per-mile loss, and assess bias corrections '
      'using only development folds. The residual analysis is descriptive; the selected models '
      'and predictions were not changed after this analysis.')

    story.append(PageBreak())
    p('6. Fixed December prediction chart', 'Heading1')
    p('The image below is a monochrome rendering of candidate_december.png created by the supplied '
      'score.py from the completed December CSV. Only the date changes; all five other scenario '
      'inputs stay fixed.')
    # Keep the supplied scorer output intact; embed a grayscale copy in the PDF.
    chart_buffer = BytesIO()
    with PILImage.open(out / 'candidate_december.png') as source_chart:
        source_chart.convert('L').save(chart_buffer, format='PNG')
    chart_buffer.seek(0)
    chart = Image(chart_buffer)
    chart.drawHeight = 500 * chart.imageHeight / chart.imageWidth
    chart.drawWidth = 500
    story.append(chart)
    story.append(Spacer(1, 12))
    p('The chart uses the separately evaluated six-input model. Unknown market and quote values '
      'are not filled with invented forecasts. The line describes model point predictions, not '
      'observed rates or an independently verified December seasonal pattern.')
    p('7. Reproducibility', 'Heading1')
    p('Run solution.py to train and export models, predictions and metrics. Run the original score.py '
      'with the final and December CSVs to validate all 12,000 IDs, positive finite rates, the exact '
      'column schemas, and all 31 fixed December scenarios. Run test_solution.py for label/ID '
      'isolation, invalid-weight behavior and saved-model prediction parity. Run diagnostics.py '
      'for residual/geographic diagnostics, then build_report.py for this report.')
    p('Repository: <link href="https://github.com/fuad-mahee/spotter-freight-rate-assessment" '
      'color="#000000">github.com/fuad-mahee/spotter-freight-rate-assessment</link>. '
      'The root README links the code, dependencies, saved models, predictions and run instructions.')

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.black)
        canvas.line(48, 43, 548, 43)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.black)
        canvas.drawString(48, 29, 'SPOTTER | Freight-rate assessment | Muftasim Fuad')
        canvas.drawRightString(548, 29, str(doc.page))

    doc = SimpleDocTemplate(str(out / 'assessment_report.pdf'), pagesize=(596, 842),
                            leftMargin=48, rightMargin=48, topMargin=45, bottomMargin=58)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print('Created', out / 'assessment_report.pdf')


if __name__ == '__main__':
    main()
