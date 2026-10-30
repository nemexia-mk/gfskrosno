(function(){'use strict';
 var root=document.documentElement,observer,timer,main,button,ended=false;
 function finish(){if(ended)return;ended=true;root.classList.remove('mk-home-pending');if(main)main.removeAttribute('aria-busy');if(observer)observer.disconnect();clearTimeout(timer);}
 // Fails open: without JavaScript, or if boot fails, the server content remains usable.
 try{
  root.classList.add('mk-home-pending');
  timer=setTimeout(finish,15000);
  function start(){try{
   if(ended)return;
   main=document.getElementById('main');button=document.getElementById('mk-load-basic');
   if(!main||!document.getElementById('mk-home-loader')){finish();return;}
   main.setAttribute('aria-busy','true');
   if(button)button.addEventListener('click',finish);
   function check(){if(main.children.length&&!main.querySelector('.mk-seo-content'))finish();}
   observer=new MutationObserver(check);observer.observe(main,{childList:true});check();
   window.addEventListener('pagehide',finish,{once:true});
  }catch(e){finish();}}
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start,{once:true});else start();
 }catch(e){finish();}
})();
