<?php
/** Plugin Name: Meteo Krosno — wesprzyj serwis
 * Version: 20261030e
 */
defined('ABSPATH') || exit;
function mkcoffee_html($html) {
 if(strpos($html,'id="mk-coffee-support"')!==false||stripos($html,'<footer')===false)return $html;
 $style=<<<'CSS'
<style id="mk-coffee-style">
.header-inner>.mk-header-coffee{display:inline-flex;align-items:center;justify-content:center;gap:8px;flex-shrink:0;margin-left:auto;margin-right:auto;padding:12px 18px;min-height:48px;box-sizing:border-box;border:1px solid #58bfa4;border-radius:14px;background:#20483f;color:#edfff8!important;text-decoration:none;font-size:14px;font-weight:700;white-space:nowrap}
.header-inner>.mk-header-coffee:hover{background:#2b5b50}.header-inner>.mk-header-coffee:focus-visible{outline:3px solid #86e6cc;outline-offset:4px}.header-inner>.desktop-nav{margin-left:0}
@media(max-width:1100px){.header .header-inner{gap:14px}.header-inner>.mk-header-coffee{padding:10px 12px;margin-right:0}}
@media(max-width:700px){.header .header-inner{gap:8px}.header .mk-brand-logo{width:clamp(112px,35vw,180px);min-width:0;flex:0 1 auto}.header .mk-brand-logo svg{width:100%;height:auto}.header-inner>.mk-header-coffee{font-size:12px;min-height:40px;padding:8px 10px;gap:5px}.header .atlas-link{display:none}}

.mk-coffee{max-width:1180px;margin:36px auto;padding:0 20px;box-sizing:border-box;color:#edf7fb;font-family:inherit}
.mk-coffee-box{display:flex;align-items:center;gap:32px;background:#102631;border:1px solid #2c5360;border-radius:24px;padding:28px;box-sizing:border-box}
.mk-coffee-copy{flex:1;min-width:0}.mk-coffee h2{color:#edf7fb!important;font-size:clamp(21px,3vw,28px);line-height:1.3;margin:0 0 12px}.mk-coffee p{font-size:15px;line-height:1.65;color:#b3d0dd;margin:0 0 18px}
.mk-coffee-options{display:flex;gap:10px;flex-wrap:wrap}.mk-coffee-option{display:flex;align-items:center;justify-content:center;gap:10px;min-height:48px;min-width:108px;padding:10px 16px;background:#20483f;border:1px solid #4cb99e;border-radius:14px;color:#d9fff3!important;text-decoration:none;font-weight:700;box-sizing:border-box}
.mk-coffee-option:hover{background:#2b5b50}.mk-coffee a:focus-visible{outline:3px solid #86e6cc;outline-offset:4px}.mk-coffee-option svg{width:25px;height:25px;flex:none}
.mk-coffee-qr{flex:none;text-align:center;color:#b3d0dd;text-decoration:none;font-size:12px}.mk-coffee-qr img{display:block;width:180px;height:auto;object-fit:contain;padding:0;background:transparent;border-radius:14px;box-sizing:border-box;margin:0 auto 8px}
.mk-coffee small{display:block;color:#91aebc;margin-top:14px;line-height:1.5;font-size:12px}
@media(max-width:600px){.mk-coffee-box{flex-direction:column;gap:22px;padding:24px 18px}.mk-coffee-copy{width:100%}.mk-coffee-options{gap:8px}.mk-coffee-option{flex:1;min-width:0;padding:10px 8px}.mk-coffee-qr img{width:160px;height:auto}}
</style>
CSS;
 $html=preg_replace_callback('~</head>~i',fn($m)=>$style.$m[0],$html,1);
 $headerButton='<a class="mk-header-coffee" href="https://buycoffee.to/meteokrosno" target="_blank" rel="noopener noreferrer" aria-label="Postaw kawę Meteo Krosno przez BuyCoffee"><span aria-hidden="true">☕</span> Postaw kawę</a>';
 $html=preg_replace_callback('~<a\b[^>]*class=["\'][^"\']*\bmk-brand-logo\b[^"\']*["\'][^>]*>.*?</a>~is',fn($m)=>$m[0].$headerButton,$html,1);
 $card='<aside class="mk-coffee" id="mk-coffee-support" aria-labelledby="mk-coffee-title"><div class="mk-coffee-box"><div class="mk-coffee-copy"><h2 id="mk-coffee-title">Postaw kawę Meteo Krosno</h2><p>Tworzymy lokalne prognozy i udostępniamy pomiary bez reklam. Jeśli korzystasz z Meteo Krosno, możesz wesprzeć rozwój serwisu i naszych stacji.</p><div class="mk-coffee-options" aria-label="Wybierz kwotę wsparcia">';
 foreach(['small'=>10,'medium'=>20,'large'=>35] as $size=>$price){
  $card.='<a class="mk-coffee-option" href="https://buycoffee.to/meteokrosno?coffeeSize='.$size.'" target="_blank" rel="noopener noreferrer" aria-label="Wesprzyj Meteo Krosno kwotą '.$price.' zł przez BuyCoffee"><svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M7 12h16v8a7 7 0 0 1-7 7h-2a7 7 0 0 1-7-7zm16 1h2a4 4 0 0 1 0 8h-2M4 28h24M11 8V4m5 4V2m5 6V4" stroke-linecap="round" stroke-linejoin="round"/></svg><span>'.$price.' zł</span></a>';
 }
 $card.='</div><small>Dobrowolne wsparcie przez BuyCoffee. Link otworzy się w nowej karcie.</small></div><a class="mk-coffee-qr" href="https://buycoffee.to/meteokrosno" target="_blank" rel="noopener noreferrer"><img src="/wp-content/themes/meteo-krosno-modern/assets/qr-buycoffee.png" width="180" height="229" loading="lazy" decoding="async" alt="Kod QR do wsparcia Meteo Krosno w BuyCoffee"><span>Zeskanuj i postaw kawę</span></a></div></aside>';
 return preg_replace_callback('~<footer\b~i',fn($m)=>$card.$m[0],$html,1);
}
add_action('template_redirect',function(){
 if(is_admin()||is_feed()||(function_exists('mkseo_internal')&&mkseo_internal())||isset($_GET['mk_api']))return;
 ob_start('mkcoffee_html');
},-1001);
