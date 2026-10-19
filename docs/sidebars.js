// This file owns the documentation hierarchy; asset generation never modifies it.
module.exports = {
  docs: [
    'about',
    'installation',
    'usage',
    { type: 'category', label: '📁 Outputs', collapsed: false,
      items: ['outputs/csv', 'outputs/explanations', 'outputs/analysis-and-qc'] },
    { type: 'category', label: '🧠 Advanced', collapsed: false,
      items: [
        'advanced/offline-usage',
        'advanced/skipping-steps-during-re-execution',
        'advanced/advanced-extraction',
        'advanced/code-changes'
      ] },
    { type: 'category', label: 'Test-retest reliability', collapsed: false,
      link: { type: 'doc', id: 'reliability/overview' },
      items: ['reliability/macula-centered', 'reliability/od-centered'] },
    { type: 'category', label: 'Biomarkers', collapsed: false,
      link: { type: 'doc', id: 'biomarkers/overview' },
      items: [
        { type: 'doc', label: 'Bifurcation features', id: 'biomarkers/bifurcation' },
        { type: 'category', label: 'CREs, diameters, and AVR',
          items: ['biomarkers/cre', 'biomarkers/caliber'] },
        { type: 'doc', label: 'Tortuosity', id: 'biomarkers/tortuosity' },
        { type: 'doc', label: 'Optic disc morphology', id: 'biomarkers/optic-disc' },
        { type: 'category', label: 'Image quality',
          items: ['biomarkers/sharpness', 'biomarkers/luminance', 'biomarkers/variance-of-laplacian'] },
        { type: 'category', label: 'Other',
          items: ['biomarkers/sparsity', 'biomarkers/fractal-dimension',
            'biomarkers/vascular-density', 'biomarkers/background-lab', 'biomarkers/temporal-angle'] },
      ] },
  ],
};
