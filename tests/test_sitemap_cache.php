<?php
define('ABSPATH', __DIR__);
$_SERVER['REQUEST_URI'] = '/page-sitemap.xml';
function add_action($name, $callback, $priority = 10) {}
function add_filter($name, $callback, $priority = 10) {}
require __DIR__ . '/../site/mu-plugins/mk-public-seo.php';
if (!defined('DONOTCACHEPAGE') || DONOTCACHEPAGE !== true || !defined('DONOTMINIFY') || DONOTMINIFY !== true) {
    throw new RuntimeException('XML sitemaps must bypass HTML caching and minification.');
}
echo "Sitemap cache exclusion passed.\n";
