const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('site/app.js','utf8');
const begin=source.indexOf("let verifyStation='turaszowka'"),end=source.indexOf('\n}',source.indexOf('async function verification(){',begin))+2;
const elements=new Map(),el=id=>{if(!elements.has(id))elements.set(id,{});return elements.get(id);};
const valid=v=>typeof v==='number'&&Number.isFinite(v);
const fmt=(v,n=1)=>valid(v)?v.toFixed(n):'—';
let response,urls=[],downloads=[];
const ctx={console,Date,Intl,document:{},main:{},$:el,valid,fmt,signed:v=>valid(v)?String(v):'—',localStamp:t=>new Date(t).toISOString(),head:()=>'<h1>Sprawdzalność</h1>',featureNav:()=>'',stat:(a,b,c)=>a+' '+b+' '+c,api:(k,q)=>'/'+k+q,csvCell:v=>'"'+String(v??'').replaceAll('"','""')+'"',download:(...x)=>downloads.push(x),fetch:async u=>{urls.push(u);return {ok:true,json:async()=>response};}};
vm.createContext(ctx);vm.runInContext(source.slice(begin,end),ctx);
const time=Date.parse('2026-10-29T12:00:00Z');
const temperature={runs:3,observations:7,pairs:[{time,run:time/1000-86400,lead:24,forecast:8,observed:7,error:1,observedAt:time}],stats:[{bin:'6–24',n:1,mae:1,rmse:1,bias:1,within2:100}]};
const rain={...temperature,metric:'rain',pairs:[{...temperature.pairs[0],start:time-21600000,observedStartAt:time-21600000}],stats:[{bin:'6–24',n:1,mae:1,rmse:1,bias:1,pod:0,far:null,csi:0,hits:0,misses:1,falseAlarms:0,correctDry:0}]};
(async()=>{
 response=temperature;await ctx.verification();
 assert.match(ctx.main.innerHTML,/verify-metric/);assert.match(el('#verify-results').innerHTML,/°C/);assert.match(el('#verify-results').innerHTML,/Błąd ≤ 2°C/);assert(!urls.at(-1).includes('metric=rain'));
 el('#verify-csv').onclick();assert.match(downloads.at(-1)[0],/temperatura/);
 response=rain;await el('#verify-metric').onchange({target:{value:'rain'}});await new Promise(resolve=>setImmediate(resolve));
 assert.match(urls.at(-1),/metric=rain/);assert.match(el('#verify-results').innerHTML,/mm \/ 6 h/);assert.match(el('#verify-results').innerHTML,/Wykryte opady · POD/);assert.match(el('#verify-results').innerHTML,/0%/);assert.match(el('#verify-results').innerHTML,/—%/);assert(!el('#verify-results').innerHTML.includes('Błąd ≤ 2°C'));
 el('#verify-csv').onclick();assert.match(downloads.at(-1)[0],/opady/);assert.match(downloads.at(-1)[1],/Poczatek UTC;Koniec UTC/);assert.match(downloads.at(-1)[1],/2026-10-29T06:00:00/);
 response={...rain,pairs:[],stats:[]};await ctx.verification();assert.match(el('#verify-results').innerHTML,/Archiwum opadów zbieramy/);assert(!el('#verify-results').innerHTML.includes('Ostatnie porównania'));
 response=temperature;await ctx.verification();assert.match(el('#verify-results').innerHTML,/Nie udało się pobrać/,'A temperature response must not be presented as rainfall');
 // An older request cannot overwrite a newer station/parameter selection.
 let resolveOld;ctx.fetch=u=>new Promise(resolve=>{resolveOld=()=>resolve({ok:true,json:async()=>rain});});const old=ctx.verification();
 ctx.fetch=async()=>({ok:true,json:async()=>({...rain,pairs:[],stats:[]})});await ctx.verification();const expected=el('#verify-results').innerHTML;resolveOld();await old;assert.equal(el('#verify-results').innerHTML,expected);
 console.log('Temperature/rain selection, units, scores, empty state, CSV and stale request isolation OK');
})().catch(e=>{console.error(e);process.exitCode=1;});
