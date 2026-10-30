(()=>{'use strict';
 const root=document.getElementById('mk-privacy-root');if(!root)return;
 const ready=(document.currentScript||document.querySelector('[data-mk-analytics-ready]'))?.getAttribute('data-mk-analytics-ready')==='1';
 const KEY='mk_privacy_consent_v1',AGE=180*86400000,ID='G-84KQL7ZK9H';
 const banner=document.getElementById('mk-consent-banner'),info=document.getElementById('mk-privacy-info');
 let loaded=false,lastFocus=null;
 function read(){try{const s=JSON.parse(localStorage.getItem(KEY));return s?.v===1&&typeof s.analytics==='boolean'&&s.at<=Date.now()&&Date.now()-s.at<AGE?s:null;}catch{return null;}}
 function save(analytics){const s={v:1,analytics,at:Date.now()};try{localStorage.setItem(KEY,JSON.stringify(s));}catch{}return s;}
 function clearCookies(){
  const domains=['',location.hostname,'.'+location.hostname];
  document.cookie.split(';').forEach(x=>{const name=x.split('=')[0].trim();if(!/^_ga(?:_|$)|^_gid$|^_gat/.test(name))return;domains.forEach(domain=>{document.cookie=name+'=; Max-Age=0; path=/; SameSite=Lax; Secure'+(domain?'; domain='+domain:'');});});
 }
 function enable(){
  if(!ready||loaded)return;loaded=true;window['ga-disable-'+ID]=false;
  window.dataLayer=window.dataLayer||[];window.gtag=function(){window.dataLayer.push(arguments);};
  window.gtag('consent','default',{analytics_storage:'denied',ad_storage:'denied',ad_user_data:'denied',ad_personalization:'denied'});
  window.gtag('js',new Date());window.gtag('consent','update',{analytics_storage:'granted',ad_storage:'denied',ad_user_data:'denied',ad_personalization:'denied'});
  window.gtag('config',ID,{allow_google_signals:false,allow_ad_personalization_signals:false,cookie_expires:180*86400,cookie_update:false,page_location:location.origin+location.pathname,page_referrer:document.referrer.split('?')[0].split('#')[0]});
  const script=document.createElement('script');script.async=true;script.id='mk-consented-analytics';script.src='https://www.googletagmanager.com/gtag/js?id='+ID;document.head.appendChild(script);
 }
 function close(){banner.hidden=true;lastFocus?.focus();}
 function deny(){save(false);window['ga-disable-'+ID]=true;clearCookies();close();if(loaded)location.reload();}
 document.addEventListener('click',e=>{
  if(e.target.closest('[data-mk-consent-open]')){lastFocus=e.target;banner.hidden=false;banner.querySelector('[data-mk-consent-deny]').focus();}
  if(e.target.closest('[data-mk-consent-deny]'))deny();
  if(e.target.closest('[data-mk-consent-accept]')&&ready){save(true);enable();close();}
  if(e.target.closest('[data-mk-consent-close]'))close();
  if(e.target.closest('[data-mk-privacy-info]')){lastFocus=e.target;info.hidden=false;info.querySelector('button').focus();}
  if(e.target.closest('[data-mk-privacy-close]')){info.hidden=true;lastFocus?.focus();}
 });
 root.addEventListener('keydown',e=>{if(e.key==='Escape'){close();info.hidden=true;}});
 window.addEventListener('storage',e=>{if(e.key!==KEY)return;const s=read();if(s?.analytics&&ready)enable();else if(loaded){window['ga-disable-'+ID]=true;clearCookies();location.reload();}});
 const choice=read();if(choice?.analytics&&ready)enable();else{window['ga-disable-'+ID]=true;clearCookies();}
 // Configuration without a final privacy policy keeps measurement disabled.
 if(!choice&&ready)banner.hidden=false;
})();
