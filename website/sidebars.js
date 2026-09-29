const featurePages = require('./feature-pages.json');
module.exports = {
  docs: [
    'about',
    { type: 'category', label: 'Usage', collapsed: false,
      items: ['usage/installation', 'usage/usage', 'usage/advanced'] },
    { type: 'category', label: 'Outputs', collapsed: false,
      items: ['outputs/csv', 'outputs/explanations'] },
    { type: 'category', label: 'Biomarkers & reliability', collapsed: false,
      items: featurePages.map(page => ({ type: 'doc', id: `biomarkers/${page.id}`, label: page.title })) },
  ],
};
