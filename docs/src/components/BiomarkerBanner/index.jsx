import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

// Show the first panel using its column count, independent of JPEG resolution or caption height.
const figures = [
  { label: 'Vessel caliber', page: 'caliber', source: '/assets/biomarkers/macula-centered/diam_crcl_arteries.jpg' },
  { label: 'Tortuosity', page: 'tortuosity', source: '/assets/biomarkers/macula-centered/tort_dist_max_segment_len_0p2_crcl_arteries.jpg' },
  { label: 'Fractal dimension', page: 'fractal-dimension', source: '/assets/biomarkers/macula-centered/fd_d0_crcl_arteries.jpg' },
  { label: 'Bifurcations', page: 'bifurcation', source: '/assets/biomarkers/macula-centered/bif_crcl_arteries.jpg' },
  { label: 'Vessel density', page: 'vascular-density', source: '/assets/biomarkers/macula-centered/vd_crcl_arteries.jpg' },
  { label: 'Optic disc', page: 'optic-disc', source: '/assets/biomarkers/od-centered/od.jpg', columns: 2 },
];

function FigureTile({ figure }) {
  const source = useBaseUrl(figure.source);
  return (
    <Link className={styles.tile} to={`/biomarkers/${figure.page}/`} aria-label={`Explore ${figure.label.toLowerCase()}`}>
      <img src={source} alt="" style={{width: `${(figure.columns || 3) * 104}%`}} />
      <span className={styles.label}>{figure.label}<span aria-hidden="true">↗</span></span>
    </Link>
  );
}

export default function BiomarkerBanner() {
  return (
    <section className={styles.banner} aria-label="VascX retinal biomarker gallery">
      <div className={styles.intro}>
        <span className={styles.eyebrow}>Retinalysis</span>
        <div className={styles.wordmark}>Vasc<span>X</span></div>
        <p className={styles.subtitle}>Explainable retinal vascular biomarkers</p>
        <Link className={styles.explore} to="/#features">Explore the biomarkers <span aria-hidden="true">↗</span></Link>
      </div>
      <div className={styles.mosaic}>
        {figures.map(figure => <FigureTile key={figure.page} figure={figure} />)}
      </div>
    </section>
  );
}
