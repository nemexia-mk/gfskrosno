const fs=require('fs'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync('site/mu-plugins/mk-home-loading.js','utf8');
function run({fallback=true,ready='loading',noMain=false,broken=false}={}){
 const classes=new Set(),attrs={},events={},timers=[],state={fallback};let changed,click,disconnected=false;
 const main={children:[{}],setAttribute:(k,v)=>attrs[k]=v,removeAttribute:k=>delete attrs[k],querySelector:()=>state.fallback?{}:null};
 const doc={documentElement:{classList:{add:x=>classes.add(x),remove:x=>classes.delete(x)}},readyState:ready,addEventListener:(name,fn)=>events[name]=fn,getElementById:id=>noMain?null:id==='main'?main:id==='mk-load-basic'?{addEventListener:(e,fn)=>click=fn}:{}};
 const box={document:doc,setTimeout:fn=>(timers.push(fn),timers.length),clearTimeout:()=>{},window:{addEventListener:(n,f)=>events[n]=f},MutationObserver:broken?undefined:class{constructor(fn){changed=fn}observe(){}disconnect(){disconnected=true}}};
 vm.runInNewContext(code,box);return {classes,attrs,events,timers,state,start:()=>events.DOMContentLoaded?.(),change:()=>changed?.(),click:()=>click?.(),disconnected:()=>disconnected};
}
let r=run();assert(r.classes.has('mk-home-pending'));r.start();assert.equal(r.attrs['aria-busy'],'true');assert(r.classes.has('mk-home-pending'));r.state.fallback=false;r.change();assert(!r.classes.size);assert(!r.attrs['aria-busy']);assert(r.disconnected());
r=run();r.start();r.click();assert(!r.classes.size);r=run();r.start();r.timers[0]();assert(!r.classes.size);assert(!r.attrs['aria-busy']);
r=run({fallback:false,ready:'complete'});assert(!r.classes.size);r=run({noMain:true});r.start();assert(!r.classes.size);r=run({broken:true});r.start();assert(!r.classes.size);
r=run();r.start();r.events.pagehide();assert(!r.classes.size);
console.log('PASS: pending, real render, cached fast render, manual fallback, timeout, missing main, script failure, page lifecycle');
