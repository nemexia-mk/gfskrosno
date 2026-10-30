const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync('site/publication-assets/consent-20261029.js','utf8');
function setup(choice,ready=true){
 const listeners={},stored=new Map(choice?[['mk_privacy_consent_v1',JSON.stringify(choice)]]:[]),scripts=[],banner={hidden:true,querySelector:()=>({focus(){}})},info={hidden:true,querySelector:()=>({focus(){}})};
 let reloads=0;
 const root={addEventListener:(name,fn)=>listeners[name]=fn};
 const document={addEventListener:(name,fn)=>listeners[name]=fn,currentScript:{getAttribute:()=>ready?'1':'0'},getElementById:id=>({'mk-privacy-root':root,'mk-consent-banner':banner,'mk-privacy-info':info})[id],cookie:'_ga=test; _ga_OLD=test; essential=ok',referrer:'https://example.com/?email=private#fragment',createElement:()=>({}),head:{appendChild:x=>scripts.push(x)}};
 const window={addEventListener:(name,fn)=>listeners[name]=fn};
 const ctx={document,window,location:{hostname:'meteo-krosno.pl',origin:'https://meteo-krosno.pl',pathname:'/prognoza/',reload:()=>reloads++},localStorage:{getItem:k=>stored.get(k)??null,setItem:(k,v)=>stored.set(k,v)},Date,JSON,console};vm.createContext(ctx);vm.runInContext(source,ctx);
 const click=selector=>listeners.click({target:{closest:s=>s===selector,focus(){}}});
 return {ctx,window,scripts,banner,info,stored,click,reloads:()=>reloads};
}
let s=setup();assert.equal(s.scripts.length,0);assert.equal(s.banner.hidden,false);s.click('[data-mk-consent-deny]');assert.equal(s.scripts.length,0);assert.equal(JSON.parse(s.stored.get('mk_privacy_consent_v1')).analytics,false);
s=setup();s.click('[data-mk-consent-accept]');s.click('[data-mk-consent-accept]');assert.equal(s.scripts.length,1);assert.match(s.scripts[0].src,/G-84KQL7ZK9H/);
let commands=s.window.dataLayer.map(x=>Array.from(x));assert.equal(commands[0][0],'consent');assert.equal(commands[0][2].analytics_storage,'denied');assert.equal(commands.find(x=>x[1]==='update')[2].ad_storage,'denied');const config=commands.find(x=>x[0]==='config')[2];assert.equal(config.page_location,'https://meteo-krosno.pl/prognoza/');assert.equal(config.page_referrer,'https://example.com/');
s.click('[data-mk-consent-deny]');assert.equal(s.reloads(),1);assert.equal(s.window['ga-disable-G-84KQL7ZK9H'],true);
s=setup({v:1,analytics:false,at:Date.now()});assert.equal(s.scripts.length,0);assert.equal(s.banner.hidden,true);s.click('[data-mk-consent-open]');assert.equal(s.banner.hidden,false);
s=setup({v:1,analytics:true,at:Date.now()-181*86400000});assert.equal(s.scripts.length,0);assert.equal(s.banner.hidden,false);
s=setup({v:1,analytics:true,at:Date.now()});assert.equal(s.scripts.length,1);
s=setup({v:1,analytics:true,at:Date.now()},false);assert.equal(s.scripts.length,0);s.click('[data-mk-consent-accept]');assert.equal(s.scripts.length,0);
console.log('Consent: zero Analytics before choice/after denial, explicit grant, one loader, withdrawal, expiry, and policy readiness OK');

s=setup(null,false);s.click('[data-mk-privacy-info]');assert.equal(s.info.hidden,false);assert.equal(s.scripts.length,0);s.click('[data-mk-privacy-close]');assert.equal(s.info.hidden,true);console.log('PASS: privacy link outside footer opens dialog without enabling analytics');
