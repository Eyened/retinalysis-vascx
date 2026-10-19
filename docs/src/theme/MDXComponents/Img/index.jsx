import React, {useEffect, useId, useRef, useState} from 'react';
import {createPortal} from 'react-dom';
import {useLocation} from '@docusaurus/router';
import OriginalImg from '@theme-original/MDXComponents/Img';
import styles from './styles.module.css';

function ExpandableImage(props) {
  const [open, setOpen] = useState(false);
  const dialog = useRef(null);
  const trigger = useRef(null);
  const titleId = useId();
  const label = props.alt || 'Figure';

  useEffect(() => {
    if (!open) return undefined;
    const element = dialog.current;
    const previousOverflow = document.body.style.overflow;
    element.showModal();
    document.body.style.overflow = 'hidden';
    return () => {
      element.close();
      document.body.style.overflow = previousOverflow;
      trigger.current?.focus({preventScroll: true});
    };
  }, [open]);

  return <>
    <button ref={trigger} type="button" className={styles.trigger}
      aria-label={`Expand ${label}`} aria-haspopup="dialog"
      onClick={() => setOpen(true)}>
      <OriginalImg {...props} />
      <span className={styles.hint} aria-hidden="true">Expand figure ↗</span>
    </button>
    {open && createPortal(
      <dialog ref={dialog} className={styles.dialog} aria-labelledby={titleId}
        onCancel={event => { event.preventDefault(); setOpen(false); }}
        onClick={event => { if (event.target === event.currentTarget) setOpen(false); }}>
        <div className={styles.content}>
          <div className={styles.toolbar}>
            <span id={titleId} className={styles.title}>{label}</span>
            <button type="button" autoFocus onClick={() => setOpen(false)} aria-label="Close figure">Close ×</button>
          </div>
          <div className={styles.viewport}>
            <img src={props.src} alt={props.alt || 'Figure'}
              className={styles.fitted} />
          </div>
        </div>
      </dialog>, document.body)}
  </>;
}

export default function MDXImg(props) {
  // Docusaurus fingerprints image URLs, so select by document route.
  const {pathname} = useLocation();
  return typeof props.src === 'string' && /\/(biomarkers|reliability)\//.test(pathname)
    ? <ExpandableImage {...props} /> : <OriginalImg {...props} />;
}
