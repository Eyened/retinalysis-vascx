module.exports = async function createConfig() {
return {
  title: 'retinalysis-vascx',
  url: 'https://eyened.github.io',
  baseUrl: '/retinalysis-vascx/',
  organizationName: 'Eyened',
  projectName: 'retinalysis-vascx',
  trailingSlash: true,
  onBrokenLinks: 'throw',
  markdown: { format: 'detect', hooks: { onBrokenMarkdownLinks: 'throw' } },
  i18n: { defaultLocale: 'en', locales: ['en'] },
  presets: [['classic', {
    docs: { path: 'docs', routeBasePath: '/', sidebarPath: require.resolve('./sidebars.js'),
      remarkPlugins: [(await import('remark-math')).default],
      rehypePlugins: [[(await import('rehype-katex')).default, { strict: 'error', throwOnError: true }]],
    },
    blog: false,
    pages: false,
    theme: { customCss: './src/css/custom.css' },
  }]],
  themeConfig: {
    colorMode: { defaultMode: 'light', disableSwitch: true },
    navbar: { title: 'retinalysis-vascx', items: [
      { href: 'https://eyened.github.io/retinalysis-vascx-app/', label: 'VascX App', position: 'right' },
      { href: 'https://eyened.github.io/retinalysis/', label: 'Retinalysis', position: 'right' },
      { href: 'https://github.com/Eyened/retinalysis-vascx', label: 'GitHub', position: 'right' },
    ] },
  },
};

};
