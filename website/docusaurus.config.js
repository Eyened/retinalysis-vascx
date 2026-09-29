module.exports = {
  title: 'VascX',
  url: 'https://eyened.github.io',
  baseUrl: '/retinalysis-vascx/',
  organizationName: 'Eyened',
  projectName: 'retinalysis-vascx',
  trailingSlash: true,
  onBrokenLinks: 'throw',
  markdown: { format: 'md', hooks: { onBrokenMarkdownLinks: 'throw' } },
  i18n: { defaultLocale: 'en', locales: ['en'] },
  presets: [['classic', {
    docs: { path: 'docs', routeBasePath: '/', sidebarPath: require.resolve('./sidebars.js') },
    blog: false,
    pages: false,
    theme: { customCss: './src/css/custom.css' },
  }]],
  themeConfig: {
    colorMode: { defaultMode: 'light', disableSwitch: true },
    navbar: { title: 'VascX', items: [
      { href: 'https://eyened.github.io/retinalysis/', label: 'Retinalysis', position: 'right' },
      { href: 'https://github.com/Eyened/retinalysis-vascx', label: 'GitHub', position: 'right' },
    ] },
  },
};
