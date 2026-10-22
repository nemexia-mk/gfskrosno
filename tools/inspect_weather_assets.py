import ftplib,os,io,re
root='/meteo-krosno.pl'
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=40) as ftp:
    for path in (root+'/wp-content/themes/meteo-krosno-modern/index.php',root+'/.htaccess'):
        out=io.BytesIO();ftp.retrbinary('RETR '+path,out.write)
        print('SOURCE',path)
        for line in out.getvalue().decode('utf-8').splitlines():
            if re.search(r'app\.js|filemtime|supercache|cache/|Cache|RewriteRule|RewriteCond',line): print(line[:500])
    for path in (root+'/wp-content/cache',root+'/wp-content/cache/supercache/meteo-krosno.pl',root+'/wp-content/themes/meteo-krosno-modern/cache'):
        try: print('CACHE',path,ftp.nlst(path)[:35])
        except ftplib.error_perm: print('CACHE_MISSING',path)
