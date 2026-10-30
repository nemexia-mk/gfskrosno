<?php
/** Plugin Name: Meteo Krosno — aplikacja Android
 * Version: 1.0.3
 */
defined('ABSPATH') || exit;
function mkandroid_release(){return json_decode(file_get_contents(__DIR__.'/mk-android-release.json'),true);}
function mkandroid_path(){return trim((string)parse_url($_SERVER['REQUEST_URI']??'/',PHP_URL_PATH),'/');}
add_filter('mkseo_routes',function($routes){
 $routes['aplikacja']=['Aplikacja Meteo Krosno na Androida — pobierz APK','Aplikacja Meteo Krosno','Pobierz bezpłatną aplikację Meteo Krosno na Androida. Lokalna pogoda, pomiary stacji, prognoza GFS i powiadomienia. Wymagania i instrukcja instalacji APK.','Pogoda Krosna i okolic w telefonie. Bezpłatna aplikacja na Androida od Meteo Krosno.'];return $routes;
});
function mkandroid_button($label='Pobierz na Androida'){
 $r=mkandroid_release();return '<a class="mk-apk-button" href="'.esc_url($r['url']).'" download="'.esc_attr($r['filename']).'">↓ '.esc_html($label).'</a>';
}
function mkandroid_page(){
 $r=mkandroid_release();$meta='Wersja '.esc_html($r['version']).' · Android 8.0 lub nowszy · '.number_format($r['bytes']/1000,0,',',' ').' kB';
 $h='<div class="mk-app-page"><section class="mk-app-hero"><div><p class="mk-app-eyebrow">METEO KROSNO · ANDROID</p><h1>Aplikacja Meteo Krosno</h1><p class="mk-app-lead">Lokalna pogoda zawsze pod ręką.</p><p>Pomiary z Turaszówki i Leśniówki, prognoza dla Krosna i ostrzeżenia pogodowe w jednej aplikacji. Bezpłatnie, bez konta i bez reklam w aplikacji.</p><a class="mk-app-textlink" href="#funkcje">Zobacz, co potrafi ↓</a></div><div class="mk-app-download" id="pobierz"><span class="mk-app-phone" aria-hidden="true">☀</span><h2>Pobierz aplikację</h2>'.mkandroid_button('Pobierz APK 1.0.0').'<p class="mk-app-meta">'.$meta.'</p><p>Instalacja bezpośrednio z naszej strony. Aplikacja nie jest obecnie dystrybuowana przez Google Play.</p><a class="mk-app-textlink" href="#instalacja">Jak zainstalować? →</a></div></section>';
 $features=[['Pomiary naszych stacji','Temperatura, wilgotność, wiatr i porywy, opady, ciśnienie oraz czas pomiaru. Wybieraj Turaszówkę lub Leśniówkę.'],['Prognoza godzinowa i dzienna','Prognoza GFS dla Krosna z temperaturą, opadami, ciśnieniem i wiatrem. Porównuj nadchodzące godziny oraz dni.'],['Powiadomienia, które wybierasz','Poranna prognoza o ustalonej godzinie, nowe ostrzeżenia IMGW i alerty z pomiarów naszych stacji.'],['Meteorogram i mapy','Meteorogram GFS, radar opadów i mapa wyładowań dostępne z aplikacji. Te widoki wymagają połączenia z internetem.'],['Klimat i historia pogody','Podsumowania miesięcy, wykresy temperatury i opadów oraz ciekawostki z wieloletnich pomiarów Krosna.'],['Meteo Krosno w Twoim telefonie','Ciemny interfejs, polskie godziny i oficjalne logo serwisu. Autor: Meteo Krosno.']];
 $h.='<section id="funkcje"><h2>Pogoda i klimat w jednym miejscu</h2><div class="mk-app-features">';foreach($features as $f)$h.='<article class="mk-app-tile"><h3>'.esc_html($f[0]).'</h3><p>'.esc_html($f[1]).'</p></article>';$h.='</div></section>';
 $h.='<section class="mk-app-panel" id="instalacja"><h2>Jak zainstalować aplikację?</h2><ol><li>Pobierz plik APK przyciskiem na tej stronie.</li><li>Otwórz pobrany plik. Jeśli Android o to poprosi, zezwól używanej przeglądarce na instalowanie aplikacji z tego źródła.</li><li>Wybierz „Zainstaluj”, a następnie otwórz Meteo Krosno. Po instalacji możesz cofnąć zgodę przeglądarki na instalowanie aplikacji.</li><li>Jeżeli chcesz otrzymywać komunikaty, włącz zgodę na powiadomienia. W ustawieniach ⚙ wybierzesz ich rodzaje i godzinę porannej prognozy.</li></ol><p><strong>Masz już wcześniejszą wersję?</strong> Zainstaluj nowy APK na dotychczasowej aplikacji. Nie odinstalowuj jej — zachowasz swoje ustawienia.</p><p>Aktualizacje będą udostępniane na tej stronie. Odnośnik do niej znajdziesz także w ustawieniach aplikacji.</p></section>';
 $h.='<section class="mk-app-panel"><h2>Warto wiedzieć</h2><p>Aktualne pomiary, prognozy i mapy wymagają internetu. Gdy odczyt jest niedostępny, aplikacja może pokazać ostatnio zapisane dane z ich rzeczywistym czasem.</p><p>Alert wiatru dotyczy przekroczenia 70 km/h i jest wysyłany najwyżej raz na godzinę. Informacja o przekroczeniu 20 mm opadu w bieżącym dniu jest wysyłana najwyżej raz dziennie. Android może opóźniać pracę w tle i dostarczenie powiadomień.</p><p>Aplikacja nie wymaga dostępu do lokalizacji, kontaktów, mikrofonu ani plików użytkownika. <button type="button" class="mk-app-textlink" data-mk-privacy-info>Informacje o prywatności</button>.</p></section>';
 $h.='<section class="mk-app-panel"><h2>Plik i wydanie</h2><p>'.$meta.' · Autor: Meteo Krosno · Wydanie z 30 października 2026 r.</p><details><summary>Suma kontrolna SHA-256</summary><code class="mk-app-hash">'.esc_html($r['sha256']).'</code></details><p>'.mkandroid_button('Pobierz aplikację').'</p></section></div>';return $h;
}
function mkandroid_html($html){
 if(stripos($html,'<html')===false||strpos($html,'id="mk-android-style"')!==false)return $html;
 $html=str_replace('consent-20261029.js','consent-20261029.js?v=20261030-click',$html);
 $style='<style id="mk-android-style">'.file_get_contents(__DIR__.'/mk-android-release.css').'</style>';
 $html=preg_replace_callback('~</head>~i',fn($m)=>$style.$m[0],$html,1);
 // Link appears in both the desktop and mobile About menus.
 $html=preg_replace('~(<summary>O serwisie(?:⌄)?</summary>\s*<div>)~u','$1<a href="/aplikacja/">Aplikacja Android</a>',$html);
 $html=preg_replace_callback('~<footer\b.*?</footer>~is',fn($m)=>str_replace('</footer>','<div class="mk-app-footer"><a href="/aplikacja/">Aplikacja Meteo Krosno na Androida</a></div></footer>',$m[0]),$html,1);
 if(mkandroid_path()==='aplikacja'){
  $html=preg_replace_callback('~(<main\b[^>]*\bid=["\']main["\'][^>]*>).*?(</main>)~is',fn($m)=>$m[1].mkandroid_page().$m[2],$html,1);
  // This page is server-rendered. Keep consent and plugin scripts, replace only the SPA bootstrap.
  $html=preg_replace('~<script\b[^>]*\bsrc=["\'][^"\']*/assets/app\.js(?:\?[^"\']*)?["\'][^>]*>\s*</script>~i','',$html);
  $boot=<<<'JS'
<script data-no-optimize="1">document.addEventListener('DOMContentLoaded',function(){var b=document.getElementById('menu-toggle'),n=document.getElementById('mobile-nav');if(b&&n)b.addEventListener('click',function(){n.hidden=!n.hidden;b.setAttribute('aria-expanded',String(!n.hidden));b.setAttribute('aria-label',n.hidden?'Otwórz menu':'Zamknij menu');});var d=document.getElementById('today-label'),y=document.getElementById('copyright-year');if(d)d.textContent=new Date().toLocaleDateString('pl-PL',{timeZone:'Europe/Warsaw',weekday:'long',day:'numeric',month:'long',year:'numeric'});if(y)y.textContent=new Date().getFullYear();});</script>
JS;
  $html=str_replace('</body>',$boot.'</body>',$html);
 }elseif(mkandroid_path()===''){
  $promo='<aside class="mk-app-promo" id="mk-android-promo"><div><p class="mk-app-eyebrow">METEO KROSNO NA ANDROIDA</p><h2>Pogoda Krosna w Twoim telefonie</h2><p>Pomiary naszych stacji, prognoza i powiadomienia zawsze pod ręką.</p></div><div class="mk-app-promo-actions">'.mkandroid_button('Pobierz aplikację').'<a class="mk-app-textlink" href="/aplikacja/">Poznaj aplikację i instrukcję instalacji →</a></div></aside>';
  if(strpos($html,'id="mk-coffee-support"')!==false)$html=preg_replace_callback('~<aside\b[^>]*id="mk-coffee-support"[^>]*>~',fn($m)=>$promo.$m[0],$html,1);
  else $html=preg_replace_callback('~<footer\b~i',fn($m)=>$promo.$m[0],$html,1);
 }
 $navigation=<<<'JS'
<script data-no-optimize="1" id="mk-android-nav">document.addEventListener('DOMContentLoaded',function(){document.querySelectorAll('.desktop-nav,#mobile-nav').forEach(function(nav){if(nav.querySelector('a[href="/aplikacja/"]'))return;var group=Array.from(nav.querySelectorAll('details')).find(function(d){return d.querySelector('summary')?.textContent.trim()==='O serwisie';});if(!group){var about=nav.querySelector('a[href="/o-nas/"]');if(!about)return;group=document.createElement('details');if(nav.classList.contains('desktop-nav'))group.className='nav-dropdown';group.innerHTML='<summary>O serwisie</summary><div><a href="/o-nas/">O nas</a></div>';about.replaceWith(group);}var a=document.createElement('a');a.href='/aplikacja/';a.textContent='Aplikacja Android';if(location.pathname==='/aplikacja/')a.setAttribute('aria-current','page');group.querySelector('div').appendChild(a);});});</script>
JS;
 $html=str_replace('</body>',$navigation.'</body>',$html);
 return $html;
}
add_action('template_redirect',function(){
 if(is_admin()||is_feed()||(function_exists('mkseo_internal')&&mkseo_internal())||isset($_GET['mk_api'])||(defined('REST_REQUEST')&&REST_REQUEST))return;
 if(mkandroid_path()==='aplikacja'){
  global $wp_query;$wp_query->is_404=false;status_header(200);
  remove_action('template_redirect','redirect_canonical');
 }
 ob_start('mkandroid_html');
},-2100);

add_action('init',function(){if(function_exists('w3tc_flush_all')&&add_option('mkandroid_cache_release_103',time(),'','no'))w3tc_flush_all();},90);
