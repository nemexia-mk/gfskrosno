const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('site/app.js','utf8'),start=Date.parse('2026-10-22T00:00:00Z');
const ctx={mgHours:24,G:null,valid:v=>typeof v==='number'&&Number.isFinite(v),
 esc:v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;'),
 fmt:(v,n=1)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'—',
 dayLabel:()=>'',dateShort:v=>v,stormColors:['white','white','yellow','orange','red'],
 stormNames:['none','possible','weak','moderate','high']};
vm.createContext(ctx);
vm.runInContext(source.slice(source.indexOf('function mgPrecip('),source.indexOf('function mgDetails(')),ctx);
assert.equal(ctx.mgScaleColor(0,[[0,'#000000'],[10,'#ffffff']]),'#000000');
assert.equal(ctx.mgScaleColor(10,[[0,'#000000'],[10,'#ffffff']]),'#ffffff');
assert.equal(ctx.mgScaleColor(5,[[0,'#000000'],[10,'#ffffff']]),'#808080');
const rows=Array.from({length:24},(_,n)=>({time:start+n*3600000,date:'2026-10-22 '+String(n).padStart(2,'0')+':00:00',
 temp:Math.sin((n-6)/4)*7,dew:-8,groundTemp:n===7?null:Math.sin((n-6)/4)*11,
 wind:3+n,gust:n===9?null:8+n,dir:200,pressure:990+n*2,snow:0,rain:0,type:'Brak',stormLevel:0,
 cloud:100,cloudLow:100,cloudMid:50,cloudHigh:75,ceiling:200+n*80,vis:5+n,fog:10}));
const svg=ctx.mgSvg(rows,start,start+24*3600000);
assert.equal((svg.match(/data-gust-tick=/g)||[]).length,23);
assert.equal((svg.match(/data-ground-temp=/g)||[]).length,23);
assert.match(svg,/stroke="url\(#mg-temp-sign\)"/);
assert.match(svg,/fill="url\(#mg-temp-sign\)"/);
assert.match(svg,/fill="url\(#mg-fill-pressure\)"/);
assert.match(svg,/fill="url\(#mg-fill-wind\)"/);
assert.match(svg,/id="mg-temp-sign"[\s\S]*?stop-color="#d52626"[\s\S]*?stop-color="#2384e8"/);
const cloud=svg.slice(svg.indexOf('>Chmury piętra i mgła</text>'));
const ticks=[...cloud.matchAll(/<text x="194"[^>]*>([^<]+)<\/text>/g)].map(m=>m[1]);
assert.deepEqual(ticks,['0','2','4','6','8']);
assert.match(svg,/>m\/s<\/text>/);assert.match(svg,/>km\/h<\/text>/);
assert(!/NaN|undefined/.test(svg));
const missing=ctx.mgSvg(rows.map(r=>({...r,groundTemp:null})),start,start+24*3600000);
assert.equal((missing.match(/data-ground-temp=/g)||[]).length,0);
assert.match(missing,/Grunt: brak danych/);
for(const value of [-5,5]){
 const oneSign=ctx.mgSvg(rows.map(r=>({...r,temp:value,dew:value-2,groundTemp:value})),start,start+24*3600000);
 assert(!/NaN|undefined/.test(oneSign));
}
if(process.argv.includes('--preview'))fs.writeFileSync('/tmp/meteo-style-preview.html','<!doctype html><meta charset="utf-8"><body style="margin:0;background:white">'+svg+'</body>');
console.log('Styles: cloud ticks 0/2/4/6/8; ground and gust ticks; signed temperature; fixed pressure/wind colors; both wind units verified.');
