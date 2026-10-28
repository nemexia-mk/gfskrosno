<?php
/**
 * Plugin Name: Meteo Krosno – publiczne SEO
 * Description: Serwerowe metadane, treść HTML i mapa używanych adresów.
 * Version: 1.0.2
 */
if (!defined('ABSPATH')) { exit; }

function mkseo_path() {
    $path = trim((string) parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH), '/');
    $prefix = trim((string) parse_url(home_url('/'), PHP_URL_PATH), '/');
    if ($prefix && strpos($path, $prefix . '/') === 0) { $path = substr($path, strlen($prefix) + 1); }
    return $path;
}

/** One explicit inventory, shared by metadata, HTML and the XML sitemap. */
function mkseo_routes() {
    return [
        '' => ['Pogoda w Krośnie – prognoza i pomiary | Meteo Krosno', 'Pogoda w Krośnie', 'Pogoda dla Krosna i powiatu krośnieńskiego: prognoza GFS, pomiary Turaszówki i Leśniówki, ostrzeżenia IMGW oraz klimat regionu.', 'Sprawdź prognozę dla Krosna, obserwacje naszych stacji i historię pogody. Meteo Krosno to społeczny serwis poświęcony pogodzie i klimatowi Krosna oraz okolic.'],
        'prognoza' => ['Prognoza pogody Krosno – godzinowa GFS | Meteo Krosno', 'Prognoza GFS dla Krosna', 'Godzinowa prognoza GFS dla Krosna: temperatura, opady, wiatr, porywy i zachmurzenie. Terminy w czasie polskim, z wyborem przebiegu modelu.', 'Prognoza modelu GFS pokazuje temperaturę na wysokości 2 metrów, opad, wiatr, ciśnienie i zachmurzenie. Godziny prezentujemy w strefie Europe/Warsaw. Wyniki modelu są prognozą, a nie pomiarem stacji; częściowe doby i braki danych wymagają odrębnej interpretacji.'],
        'meteorogram' => ['Meteorogram GFS dla Krosna – 24–120 godzin | Meteo Krosno', 'Meteorogram GFS dla Krosna', 'Meteorogram GFS dla Krosna na 24, 48, 72 lub 120 godzin. Temperatura, opad, wiatr i chmury na wykresie z osią czasu polskiego oraz UTC.', 'Meteorogram zestawia parametry prognozy GFS na wspólnej osi czasu. Zakres zaczyna się od pierwszego terminu wybranego przebiegu modelu. Wykres interaktywny pozwala porównać temperaturę, opad, wilgotność, ciśnienie, wiatr i zachmurzenie; luki w danych nie oznaczają wartości zerowej.'],
        'sprawdzalnosc' => ['Sprawdzalność prognoz GFS dla Krosna | Meteo Krosno', 'Sprawdzalność prognoz GFS', 'Porównanie temperatury prognozowanej przez GFS z pomiarami Turaszówki i Leśniówki. Błędy MAE, RMSE i obciążenie dla wyprzedzeń od 6 do 72 godzin.', 'Porównujemy zapisane prognozy GFS z rzeczywistymi pomiarami lokalnych stacji. Wyniki dotyczą dostępnych par prognoza–pomiar i są podzielone według wyprzedzenia prognozy. MAE oznacza średni błąd bezwzględny, RMSE mocniej uwzględnia duże błędy, a obciążenie pokazuje kierunek odchylenia prognoz.'],
        'miesiac-na-zywo' => ['Pogoda w Krośnie – bilans bieżącego miesiąca | Meteo Krosno', 'Miesiąc na żywo – klimat Krosna', 'Bieżący bilans IMGW Krosno: średnia temperatura, opady, usłonecznienie i przymrozki. Porównanie z normami 1991–2020 i informacja o kompletności danych.', 'Bilans obejmuje zakończone doby z pomiarami IMGW Krosno. Temperaturę porównujemy z tymi samymi dniami wielolecia 1991–2020. Sumy opadu i usłonecznienia pokazują udział w normie pełnego miesiąca. Bieżąca doba nie wchodzi do statystyk, a brakujących pomiarów nie zastępujemy zerami.'],
        'analizator' => ['Atlas klimatu Krosna – dane IMGW od 1985 roku | Meteo Krosno', 'Atlas klimatu Krosna', 'Atlas klimatu Krosna: 22 moduły analizy danych IMGW od 1985 roku. Rekordy temperatury, opady, śnieg, przymrozki, trendy i porównania okresów.', 'Poznaj historię pogody w Krośnie na podstawie dobowych obserwacji IMGW od 1985 roku. Atlas pozwala porównywać lata i sezony, szukać rekordów, analizować fale temperatury, opady, pokrywę śnieżną i okres bezprzymrozkowy. Każdą analizę należy czytać wraz z informacją o dostępności danych i metodyce.'],
        'tego-dnia' => ['Ten dzień w historii pogody Krosna | Meteo Krosno', 'Ten dzień w historii pogody Krosna', 'Historia pogody w Krośnie dla wybranego dnia: rekordy temperatury, opad, usłonecznienie i pomiary IMGW rok po roku od 1985 roku.', 'Porównaj ten sam dzień kalendarzowy w kolejnych latach obserwacji IMGW Krosno. Zestawienie obejmuje dostępną serię pomiarów, dlatego rekord oznacza rekord tego archiwum. Wartości wieloletnie są obliczane z dostępnych obserwacji okresu 1991–2020.'],
        'stacje' => ['Mapa stacji pogodowych – Krosno i Leśniówka | Meteo Krosno', 'Mapa lokalnych stacji pogodowych', 'Mapa stacji Meteo Krosno: Turaszówka i Leśniówka. Porównaj aktualną temperaturę, wiatr i opad oraz sprawdź czas ostatniego pomiaru.', 'Lokalne stacje pokazują, jak warunki pogodowe różnią się między Turaszówką a Leśniówką. Mapa umożliwia wybór temperatury, wiatru lub opadu. Położenie znaczników jest orientacyjne. Przy porównaniu pomiarów sprawdź ich czas oraz oznaczenie świeżości danych.'],
        'mapy' => ['Radar opadów i mapy pogody – Krosno | Meteo Krosno', 'Radar opadów i mapy pogody', 'Radar opadów dla Krosna i okolic, animacja RainViewer oraz odnośniki do map IMGW, wyładowań i zdjęć satelitarnych.', 'Obserwuj opady nad Krosnem i okolicą na historycznych obrazach radarowych RainViewer. Czas klatki opisuje termin mozaiki radarowej; brak echa może oznaczać również brak pokrycia. Radar pokazuje odbiciowość, a nie sumę opadu zmierzoną przy gruncie.'],
        'stacja-turaszowka' => ['Stacja Krosno–Turaszówka – aktualna pogoda | Meteo Krosno', 'Stacja pogodowa Krosno–Turaszówka', 'Aktualne pomiary stacji Krosno–Turaszówka: temperatura, wilgotność, ciśnienie, opad, wiatr i promieniowanie słoneczne oraz wykresy archiwalne.', 'Stacja w Turaszówce rejestruje lokalną pogodę w zachodniej części Krosna. Dashboard przedstawia ostatnie dostępne pomiary i ich czas, ekstrema doby oraz wykresy. Doba opadowa trwa od 06:00 UTC do 06:00 UTC, czyli od 07:00 zimą lub 08:00 latem czasu polskiego.'],
        'stacja-lesniowka' => ['Stacja Leśniówka – aktualna pogoda | Meteo Krosno', 'Stacja pogodowa Leśniówka', 'Aktualna pogoda w Leśniówce w gminie Chorkówka: temperatura, wilgotność, wiatr, ciśnienie i opad. Pomiary lokalnej stacji i archiwalne wykresy.', 'Stacja w Leśniówce w gminie Chorkówka pomaga obserwować lokalny mikroklimat, w tym silne wychłodzenia w mrozowisku. Odczyty można porównać z Turaszówką. Dashboard pokazuje czas i świeżość pomiaru, a brak danych jest odróżniony od wartości zerowej.'],
        'wykresy' => ['Wykresy pogody i archiwum dwóch stacji | Meteo Krosno', 'Wykresy i archiwum stacji pogodowych', 'Wykresy temperatury, wilgotności, wiatru, opadu i ciśnienia ze stacji Turaszówka oraz Leśniówka. Wybierz stację, datę i zakres porównania.', 'Przeglądaj zapisane pomiary lokalnych stacji Meteo Krosno. Wykresy pozwalają prześledzić przebieg temperatury, wiatru, opadu i ciśnienia. Dostępność archiwum zależy od okresu działania stacji i kompletności zapisów.'],
        'klimat' => ['Klimat Krosna i okolic – wielolecie i mikroklimat | Meteo Krosno', 'Klimat Krosna i okolic', 'Klimat Krosna na podstawie danych IMGW i lokalnych stacji: wieloletnia temperatura, opady, różnice mikroklimatu oraz podsumowania miesięcy i lat.', 'Badamy pogodę Krosna i okolic, zestawiając długą serię IMGW z obserwacjami Turaszówki i Leśniówki. Rzeźba terenu, doliny i stoki wpływają na lokalne różnice warunków. Porównania wymagają jednoczesnych pomiarów; sama różnica odczytów nie przesądza o jej przyczynie.'],
        'klimat-miesieczny' => ['Miesięczne podsumowania klimatu Krosna | Meteo Krosno', 'Podsumowanie klimatyczne miesiąca', 'Podsumowania miesięcy w Krośnie: średnia i skrajna temperatura, opady oraz usłonecznienie na tle norm klimatycznych 1991–2020.', 'Wybierz miesiąc i rok, aby przeanalizować pomiary IMGW Krosno. Podsumowanie zestawia średnią temperaturę, wartości skrajne, sumę opadu i usłonecznienie z normami 1991–2020. Przy niepełnej serii oceniaj wyniki wraz z informacją o brakujących dobach.'],
        'klimat-roczny' => ['Roczne podsumowania klimatu Krosna | Meteo Krosno', 'Podsumowanie klimatyczne roku', 'Roczny bilans pogody w Krośnie: temperatura, opady i usłonecznienie, przebieg miesięcy oraz porównanie z normami wieloletnimi 1991–2020.', 'Roczne zestawienie pomiarów IMGW Krosno pozwala porównać przebieg miesięcy i bilans całego roku. Niepełnego bieżącego roku nie należy porównywać bezpośrednio z sumami pełnych lat. Normy odniesienia dotyczą okresu 1991–2020.'],
        'blog' => ['Artykuły o pogodzie i klimacie Krosna | Meteo Krosno', 'Artykuły i podsumowania pogody', 'Artykuły Meteo Krosno o pogodzie, klimacie i lokalnych zjawiskach oraz archiwalne podsumowania temperatury, opadów i usłonecznienia.', 'Czytaj lokalne analizy pogody i klimatu oraz podsumowania miesięcy. Archiwalne publikacje opisują okres wskazany w tytule; data publikacji może być późniejsza od analizowanego miesiąca.'],
        'category/podsumowania' => ['Podsumowania pogody w Krośnie – archiwum | Meteo Krosno', 'Archiwum podsumowań pogody w Krośnie', 'Archiwalne podsumowania miesięcy w Krośnie: temperatura, opady i usłonecznienie, wartości skrajne oraz odniesienie do norm klimatycznych.', 'Zestawienia miesięczne dokumentują przebieg pogody w Krośnie. W każdej publikacji sprawdź analizowany okres, źródło pomiarów i przyjętą normę wieloletnią.'],
        'o-nas' => ['O serwisie Meteo Krosno – lokalna pogoda i klimat', 'O serwisie Meteo Krosno', 'Poznaj Meteo Krosno: społeczny portal pogody i klimatu Krosna oraz okolic, lokalne stacje pomiarowe i analizy wieloletnich danych IMGW.', 'Meteo Krosno to społeczny serwis poświęcony pogodzie i klimatowi Krosna oraz okolic. Łączymy obserwacje lokalnych stacji, prognozy numeryczne i analizy wieloletnich danych meteorologicznych.'],
        'skrypty' => ['Narzędzia i skrypty pogodowe | Meteo Krosno', 'Narzędzia pogodowe Meteo Krosno', 'Narzędzia Meteo Krosno do odczytu i prezentacji danych pogodowych. Informacje o dostępnych skryptach, ich działaniu i zastosowaniach.', 'Poznaj narzędzia wspierające prezentację i analizę danych pogodowych. Przy korzystaniu ze skryptów zwróć uwagę na wymagany format danych, jednostki i zakres dostępnych pomiarów.'],
        'przydatne-linki' => ['Źródła danych i przydatne linki pogodowe | Meteo Krosno', 'Przydatne linki pogodowe', 'Sprawdzone źródła informacji meteorologicznych: IMGW, modele pogody, radary, zdjęcia satelitarne i mapy burz. Zestaw odnośników Meteo Krosno.', 'Sięgaj do źródeł informacji meteorologicznej, aby porównywać prognozy z obserwacjami. Aktualne ostrzeżenia sprawdzaj w IMGW-PIB, a mapy radarowe i satelitarne interpretuj zgodnie z czasem oraz charakterem prezentowanych danych.'],
        'aktualna-temperatura-w-polsce' => ['Aktualna temperatura w Polsce – mapa | Meteo Krosno', 'Mapa aktualnej temperatury w Polsce', 'Mapa temperatury w Polsce: porównaj warunki w Krośnie i innych regionach na podstawie zewnętrznej mapy obserwacji meteorologicznych.', 'Mapa umożliwia porównanie temperatury w różnych częściach Polski. Czas obserwacji i dostępność stacji zależą od zewnętrznego dostawcy; sprawdzaj termin pomiaru przed porównywaniem miejscowości.'],
        'mapa-satelitarna-polski' => ['Mapa satelitarna Polski – zachmurzenie | Meteo Krosno', 'Mapa satelitarna Polski', 'Zdjęcia satelitarne i zachmurzenie nad Polską. Obserwuj układy chmur oraz ich przemieszczanie na mapie dostępnej w Meteo Krosno.', 'Zdjęcia satelitarne pomagają śledzić zasięg i przemieszczanie zachmurzenia nad Polską. Termin obrazu jest istotny dla jego interpretacji. Sam obraz chmur nie jest pomiarem opadu przy powierzchni ziemi.'],
        '2022/08/09/klimatogram-krosno' => ['Klimatogram Krosna – temperatura i opady | Meteo Krosno', 'Klimatogram Krosna', 'Klimatogram Krosna: roczny przebieg temperatury i opadów oraz omówienie klimatu miasta na podstawie danych meteorologicznych.', 'Klimatogram pokazuje typowy przebieg temperatury i opadów w kolejnych miesiącach. Przy porównywaniu wykresów zwróć uwagę na opisany okres odniesienia i źródło obserwacji.']
    ];
}

function mkseo_internal() {
    foreach (['mk_api', 'mk_station_frame', 'mk_station_data', 'mk_station_api', 'rest_route', 'preview'] as $key) {
        if (isset($_GET[$key])) { return true; }
    }
    return (defined('REST_REQUEST') && REST_REQUEST) || (function_exists('wp_doing_ajax') && wp_doing_ajax());
}

function mkseo_excluded_post($post) {
    return !$post || $post->post_status !== 'publish' || in_array($post->post_name, ['przykladowa-strona', 'podsumowanie-testowe'], true);
}

/** Reversible one-time migration. Preserve the exact previous status privately. */
add_action('init', function () {
    if (get_option('mkseo_sample_drafted_v1')) { return; }
    $post = get_page_by_path('przykladowa-strona', OBJECT, 'page');
    if (!$post) { return; }
    if ($post->post_status !== 'draft') {
        if (!get_option('mkseo_sample_previous_status_v1')) {
            add_option('mkseo_sample_previous_status_v1', ['id' => $post->ID, 'status' => $post->post_status], '', false);
        }
        $result = wp_update_post(['ID' => $post->ID, 'post_status' => 'draft'], true);
        if (is_wp_error($result)) { return; }
    }
    add_option('mkseo_sample_drafted_v1', $post->ID, '', false);
}, 20);

// Run once after deployment, through WordPress so its cache indexes stay coherent.
add_action('init', function () {
    if (function_exists('w3tc_flush_all') && add_option('mkseo_cache_refreshed_v2', time(), '', false)) {
        w3tc_flush_all();
    }
}, 30);

function mkseo_meta() {
    $path = mkseo_path(); $routes = mkseo_routes(); $post = get_queried_object();
    $notfound = is_404() || $path === 'przykladowa-strona';
    $noindex = $notfound || is_search() || is_author() || is_date() || is_attachment() || mkseo_internal();
    if (isset($routes[$path]) && !$notfound) {
        list($title, $h1, $description, $intro) = $routes[$path];
    } elseif ($post instanceof WP_Post && !mkseo_excluded_post($post) && is_singular()) {
        $h1 = wp_strip_all_tags(get_the_title($post)); $title = $h1 . ' | Meteo Krosno';
        $description = wp_trim_words(wp_strip_all_tags(strip_shortcodes($post->post_excerpt ?: $post->post_content)), 26, '…');
        $intro = $description;
        if ($post->post_type === 'page') { $noindex = true; }
    } elseif ($notfound) {
        $title = 'Nie znaleziono strony | Meteo Krosno'; $h1 = 'Nie znaleziono strony';
        $description = 'Ten adres nie jest dostępny. Przejdź do prognozy, pomiarów i archiwum Meteo Krosno.'; $intro = $description;
    } else {
        $h1 = is_search() ? 'Wyniki wyszukiwania' : 'Archiwum Meteo Krosno';
        $title = $h1 . ' | Meteo Krosno'; $description = 'Archiwalne informacje o pogodzie i klimacie Krosna oraz okolic.'; $intro = $description; $noindex = true;
    }
    $canonical = home_url($path === '' ? '/' : '/' . $path . '/');
    if (is_singular() && $post instanceof WP_Post && !mkseo_excluded_post($post) && !isset($routes[$path])) { $canonical = get_permalink($post); }
    return compact('path', 'title', 'h1', 'description', 'intro', 'canonical', 'noindex', 'notfound');
}

function mkseo_links() {
    $html = '<nav aria-label="Pogoda i klimat Krosna"><ul>';
    foreach (mkseo_routes() as $path => $route) {
        if ($path === mkseo_path()) { continue; }
        $html .= '<li><a href="' . esc_url(home_url($path ? '/' . $path . '/' : '/')) . '">' . esc_html($route[1]) . '</a></li>';
    }
    return $html . '</ul></nav>';
}

/** Local cached data only: rendering SEO never starts an upstream weather request. */
function mkseo_forecast_html() {
    $file = get_template_directory() . '/cache/gfs-tab.csv';
    if (!is_readable($file)) { return ''; }
    $fp = fopen($file, 'rb'); if (!$fp) { return ''; }
    $head = fgetcsv($fp, 0, ',', '"', '\\');
    if (!$head || !in_array('T2M [°C]', $head, true)) { fclose($fp); return ''; }
    $rows = []; $run = null; $now = time();
    while (($r = fgetcsv($fp, 0, ',', '"', '\\')) !== false) {
        $r = array_combine($head, array_pad(array_slice($r, 0, count($head)), count($head), ''));
        $dt = DateTimeImmutable::createFromFormat('!Y-m-d H:i:s', $r['Czas'] ?? '', new DateTimeZone('UTC'));
        if (!$dt) { continue; }
        if ($run === null) { $run = $dt->getTimestamp() - (float) ($r['T+ (h)'] ?? 0) * 3600; }
        if ($dt->getTimestamp() >= $now && count($rows) < 24) { $rows[] = [$dt->getTimestamp(), $r]; }
    }
    fclose($fp);
    if (!$rows || !$run || $now - $run > 36 * 3600 || $run > $now + 3600) { return ''; }
    $html = '<section><h2>Godzinowa prognoza GFS dla Krosna</h2><p>Przebieg modelu: ' . esc_html(wp_date('d.m.Y H:i', (int) $run, new DateTimeZone('Europe/Warsaw'))) . ' czasu polskiego. To prognoza modelu, nie pomiar. Tabela obejmuje dostępne terminy.</p><div style="overflow-x:auto"><table><thead><tr><th>Termin · czas polski</th><th>Temperatura [°C]</th><th>Opad / krok [mm]</th><th>Wiatr [m/s]</th></tr></thead><tbody>';
    foreach ($rows as $row) {
        $html .= '<tr><td>' . esc_html(wp_date('d.m H:i T', $row[0], new DateTimeZone('Europe/Warsaw'))) . '</td>';
        foreach (['T2M [°C]', 'RRR [mm]', 'WSPD [m/s]'] as $key) {
            $v = str_replace(',', '.', $row[1][$key] ?? '');
            $html .= '<td>' . (is_numeric($v) ? esc_html(number_format((float) $v, 1, ',', '')) : '—') . '</td>';
        }
        $html .= '</tr>';
    }
    return $html . '</tbody></table></div></section>';
}

function mkseo_article_html($post) {
    if (mkseo_excluded_post($post) || !empty($post->post_password)) { return ''; }
    $content = apply_filters('the_content', $post->post_content);
    if (preg_match('~<body\b[^>]*>(.*?)</body>~is', $content, $match)) { $content = $match[1]; }
    $content = preg_replace('~<(script|style|head|title)\b[^>]*>.*?</\1>~is', '', $content);
    $content = preg_replace('~<(\/?)h1\b~i', '<$1h2', $content);
    return '<article class="prose">' . wp_kses_post($content) . '</article>';
}

function mkseo_content($meta) {
    $html = '<section class="card prose mk-seo-content"><h1>' . esc_html($meta['h1']) . '</h1><p>' . esc_html($meta['intro']) . '</p>';
    $post = get_queried_object();
    if (!$meta['notfound'] && is_singular() && $post instanceof WP_Post) {
        $html .= $GLOBALS['mkseo_prepared_articles'][$post->ID] ?? mkseo_article_html($post);
    }
    if (in_array($meta['path'], ['', 'prognoza', 'meteorogram'], true)) { $html .= mkseo_forecast_html(); }
    if (in_array($meta['path'], ['', 'blog', 'category/podsumowania'], true)) {
        $args = ['post_type' => 'post', 'post_status' => 'publish', 'posts_per_page' => 12, 'has_password' => false];
        if ($meta['path'] === 'category/podsumowania') { $args['category_name'] = 'podsumowania'; }
        $html .= '<section><h2>Ostatnie publikacje</h2>';
        foreach (get_posts($args) as $item) {
            if (mkseo_excluded_post($item)) { continue; }
            $html .= '<article><h3><a href="' . esc_url(get_permalink($item)) . '">' . esc_html(get_the_title($item)) . '</a></h3><p>' . esc_html(wp_trim_words(wp_strip_all_tags(strip_shortcodes($item->post_excerpt ?: $item->post_content)), 45, '…')) . '</p></article>';
        }
        $html .= '</section>';
    }
    $html .= '<p>Wykresy, przełączniki i aktualizacje pomiarów wymagają JavaScript. Podstawowe informacje i odnośniki są dostępne także bez niego.</p>';
    return $html . mkseo_links() . '</section>';
}

/** Normalize the legacy theme's competing WP/AIOSEO head output in one place. */
function mkseo_html($html) {
    if (stripos($html, '<html') === false || stripos($html, '<head') === false) { return $html; }
    $meta = mkseo_meta();
    $html = preg_replace_callback('~<head\b[^>]*>(.*?)</head>~is', function ($match) use ($meta) {
        $head = preg_replace('~<title\b[^>]*>.*?</title>~is', '', $match[1]);
        $head = preg_replace_callback('~<meta\b[^>]*>~is', function ($tag) {
            return preg_match('~(?:name|property)\s*=\s*["\'](?:description|robots|googlebot|og:[^"\']+|twitter:[^"\']+)["\']~i', $tag[0]) ? '' : $tag[0];
        }, $head);
        $head = preg_replace('~<link\b[^>]*rel\s*=\s*["\']canonical["\'][^>]*>~is', '', $head);
        $head = preg_replace('~<script\b[^>]*type\s*=\s*["\']application/ld\+json["\'][^>]*>.*?</script>~is', '', $head);
        $head .= '<title>' . esc_html($meta['title']) . '</title><meta name="description" content="' . esc_attr($meta['description']) . '"><meta name="robots" content="' . ($meta['noindex'] ? 'noindex,follow' : 'index,follow,max-image-preview:large') . '">';
        if (!$meta['notfound']) { $head .= '<link rel="canonical" href="' . esc_url($meta['canonical']) . '">'; }
        $post = get_queried_object(); $article = is_singular('post') && $post instanceof WP_Post && !mkseo_excluded_post($post);
        $image = $article ? get_the_post_thumbnail_url($post, 'large') : false;
        if (!$image) { $image = get_site_icon_url(512); }
        $fields = ['og:locale' => 'pl_PL', 'og:site_name' => 'Meteo Krosno', 'og:type' => $article ? 'article' : 'website', 'og:title' => $meta['title'], 'og:description' => $meta['description'], 'og:url' => $meta['canonical']];
        if ($image) { $fields['og:image'] = $image; }
        foreach ($fields as $name => $value) { $head .= '<meta property="' . esc_attr($name) . '" content="' . esc_attr($value) . '">'; }
        $head .= '<meta name="twitter:card" content="' . ($article && $image ? 'summary_large_image' : 'summary') . '"><meta name="twitter:title" content="' . esc_attr($meta['title']) . '"><meta name="twitter:description" content="' . esc_attr($meta['description']) . '">';
        if ($image) { $head .= '<meta name="twitter:image" content="' . esc_url($image) . '">'; }
        if (!$meta['noindex']) {
            $site = home_url('/');
            $page = ['@type' => $article ? 'Article' : 'WebPage', '@id' => $meta['canonical'] . '#page', 'url' => $meta['canonical'], 'name' => $meta['h1'], 'description' => $meta['description'], 'inLanguage' => 'pl-PL', 'isPartOf' => ['@id' => $site . '#website']];
            if ($article) { $page['headline'] = wp_strip_all_tags(get_the_title($post)); $page['datePublished'] = get_post_time('c', true, $post); $page['dateModified'] = get_post_modified_time('c', true, $post); $page['author'] = ['@type' => 'Organization', 'name' => 'Meteo Krosno', 'url' => $site]; if ($image) { $page['image'] = $image; } }
            $schema = ['@context' => 'https://schema.org', '@graph' => [['@type' => 'WebSite', '@id' => $site . '#website', 'url' => $site, 'name' => 'Meteo Krosno', 'inLanguage' => 'pl-PL'], $page]];
            $head .= '<script type="application/ld+json">' . wp_json_encode($schema, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_HEX_TAG | JSON_HEX_AMP) . '</script>';
        }
        return '<head>' . $head . '</head>';
    }, $html, 1);
    // The app progressively replaces #main; keep a single meaningful server H1.
    $replaced = 0;
    $html = preg_replace_callback('~(<main\b[^>]*\bid=["\']main["\'][^>]*>).*?(</main>)~is', function ($match) use ($meta) { return $match[1] . mkseo_content($meta) . $match[2]; }, $html, 1, $replaced);
    // Native PHP dashboards already render full data; preserve them, clarify the H1.
    if (!$replaced && in_array($meta['path'], ['stacja-turaszowka', 'stacja-lesniowka', 'klimat-miesieczny', 'klimat-roczny'], true)) {
        $html = preg_replace_callback('~(<h1\b[^>]*>)(.*?)(</h1>)~is', function ($match) use ($meta) {
            $heading = $meta['h1'];
            if (strpos($meta['path'], 'klimat-') === 0) { $heading .= ' – ' . wp_strip_all_tags($match[2]); }
            return $match[1] . esc_html($heading) . $match[3];
        }, $html, 1);
    }
    $html = preg_replace('~<noscript>\s*<main\b.*?</main>\s*</noscript>~is', '', $html);
    return $html;
}

add_action('template_redirect', function () {
    if (is_admin() || is_feed() || mkseo_internal()) { return; }
    if (mkseo_path() === 'przykladowa-strona') { global $wp_query; $wp_query->set_404(); status_header(404); nocache_headers(); }
    // Blocks and shortcodes may start their own buffers. Render outside our display
    // handler: PHP forbids starting a buffer from inside another buffer's callback.
    $post = get_queried_object();
    if (is_singular() && $post instanceof WP_Post) {
        $GLOBALS['mkseo_prepared_articles'][$post->ID] = mkseo_article_html($post);
    }
    ob_start('mkseo_html');
}, -999);
add_action('send_headers', function () {
    if (mkseo_internal()) { header('X-Robots-Tag: noindex, nofollow', true); }
});

function mkseo_sitemap_entries($kind) {
    $entries = [];
    if ($kind === 'page') {
        foreach (mkseo_routes() as $path => $route) {
            $url = home_url($path ? '/' . $path . '/' : '/');
            $entries[$url] = ['loc' => $url];
        }
    } else {
        $posts = get_posts(['post_type' => 'post', 'post_status' => 'publish', 'posts_per_page' => -1, 'has_password' => false, 'orderby' => 'ID', 'order' => 'ASC']);
        foreach ($posts as $post) {
            if (mkseo_excluded_post($post)) { continue; }
            $url = get_permalink($post);
            if (isset(mkseo_routes()[trim((string) parse_url($url, PHP_URL_PATH), '/')])) { continue; }
            $entries[$url] = ['loc' => $url, 'lastmod' => get_post_modified_time('c', true, $post)];
        }
    }
    return array_values($entries);
}

function mkseo_sitemap_xml($kind) {
    $escape = function ($s) { return htmlspecialchars($s, ENT_XML1 | ENT_QUOTES, 'UTF-8'); };
    $xml = '<?xml version="1.0" encoding="UTF-8"?>' . "\n";
    if ($kind === 'index') {
        $xml .= '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">';
        foreach (['page-sitemap.xml', 'post-sitemap.xml'] as $path) { $xml .= '<sitemap><loc>' . $escape(home_url('/' . $path)) . '</loc></sitemap>'; }
        return $xml . '</sitemapindex>';
    }
    $xml .= '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">';
    foreach (mkseo_sitemap_entries($kind) as $entry) {
        $xml .= '<url><loc>' . $escape($entry['loc']) . '</loc>';
        if (!empty($entry['lastmod'])) { $xml .= '<lastmod>' . $escape($entry['lastmod']) . '</lastmod>'; }
        $xml .= '</url>';
    }
    return $xml . '</urlset>';
}

// Intercept before competing sitemap plugins; existing public sitemap URLs stay valid.
add_action('init', function () {
    $path = mkseo_path();
    $aliases = ['sitemap_index.xml' => 'sitemap.xml', 'wp-sitemap.xml' => 'sitemap.xml', 'sitemap.rss' => 'sitemap.xml', 'addl-sitemap.xml' => 'page-sitemap.xml'];
    if (isset($aliases[$path])) { wp_safe_redirect(home_url('/' . $aliases[$path]), 301); exit; }
    if ($path === 'attachment-sitemap.xml') { status_header(410); header('X-Robots-Tag: noindex'); header('Content-Type: text/plain; charset=UTF-8'); echo 'Mapa załączników została wycofana.'; exit; }
    $types = ['sitemap.xml' => 'index', 'page-sitemap.xml' => 'page', 'post-sitemap.xml' => 'post'];
    if (!isset($types[$path])) { return; }
    status_header(200); header('Content-Type: application/xml; charset=UTF-8'); header('X-Robots-Tag: noindex, follow'); header('Cache-Control: public, max-age=300');
    echo mkseo_sitemap_xml($types[$path]); exit;
}, -900);
add_filter('robots_txt', function ($text) {
    return "User-agent: *\nDisallow: /wp-admin/\nAllow: /wp-admin/admin-ajax.php\n\nSitemap: " . home_url('/sitemap.xml') . "\n";
}, PHP_INT_MAX);
