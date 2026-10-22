const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('site/app.js','utf8');
const start=Date.parse('2026-10-22T00:00:00Z');
const ctx={mgHours:24,G:null,valid:v=>typeof v==='number'&&Number.isFinite(v),
 esc:v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;'),
 fmt:(v,n=1)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'—',
 dayLabel:()=>'',dateShort:v=>v,stormColors:['white','white','yellow','orange','red'],
 stormNames:['none','possible','weak','moderate','high']};
vm.createContext(ctx);
vm.runInContext(source.slice(source.indexOf('function mgPrecip('),source.indexOf('function mgDetails(')),ctx);
assert.equal(ctx.mgOktas(0),0);assert.equal(ctx.mgOktas(50),4);assert.equal(ctx.mgOktas(100),8);
assert.equal(ctx.mgOktas(null),null);assert.equal(ctx.mgOktas(101),null);
assert.equal(ctx.mgCloudBase({cloud:100,ceiling:0}).height,0);
assert.equal(ctx.mgCloudBase({cloud:100,ceiling:300,temp:10,dew:5}).height,300);
assert.equal(ctx.mgCloudBase({cloud:50,temp:10,dew:5}).height,625);
assert.equal(ctx.mgCloudBase({cloud:50,ceiling:999999,temp:10,dew:5}).height,625);
assert.equal(ctx.mgCloudBase({cloud:0,temp:10,dew:5,ceiling:0}).height,null);
assert.equal(ctx.mgCloudBase({cloud:50,temp:null,dew:5}).height,null);
assert.equal(ctx.mgCloudBase({cloud:50,temp:10,dew:15}).height,null);
const rows=Array.from({length:24},(_,n)=>({time:start+n*3600000,date:'2026-10-22 '+String(n).padStart(2,'0')+':00:00',
 temp:8+Math.sin(n/4)*5,dew:5,wind:2+n%4,gust:8+n%4,dir:180,pressure:1015+n/4,snow:0,rain:n%5===0?2:0,
 type:'Deszcz',stormLevel:n===12?3:0,cloud:100,cloudLow:n<6?100:50,cloudMid:50,cloudHigh:75,
 ceiling:n%3===0?null:100+n*80,vis:n<6?.2+n/10:5+n,fog:n<6?70:10}));
const svg=ctx.mgSvg(rows,start,start+24*3600000);
assert(svg.indexOf('mg-clip-base')<svg.indexOf('mg-clip-cloud'));
assert.match(svg,/data-visibility-line="true"/);assert.match(svg,/data-base-estimated="true"/);
assert.match(svg,/data-base-estimated="false"/);assert.match(svg,/0–8 oktantów/);
assert.match(svg,/Mgła/);assert.match(svg,/data-storm-placement="above-rain"/);
assert(!/NaN|undefined/.test(svg));
if(process.argv.includes('--preview'))fs.writeFileSync('/tmp/meteo-cloud-preview.html','<!doctype html><meta charset="utf-8"><body style="margin:0;background:white">'+svg+'</body>');
console.log('Cloud panel: GFS ceiling, LCL fallback, clear sky, missing data, visibility, oktas and storm markers verified.');
