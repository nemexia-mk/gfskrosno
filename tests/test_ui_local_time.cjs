const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('site/app.js','utf8');
const between=(a,b)=>source.slice(source.indexOf(a),source.indexOf(b,source.indexOf(a)));
const ctx={console,Intl,Date,valid:v=>typeof v==='number'&&Number.isFinite(v),mean:vs=>vs.filter(Number.isFinite).reduce((a,b)=>a+b,0)/vs.filter(Number.isFinite).length,min:vs=>Math.min(...vs),max:vs=>Math.max(...vs),sum:vs=>{vs=vs.filter(Number.isFinite);return vs.length?vs.reduce((a,b)=>a+b,0):null;}};
vm.createContext(ctx);
vm.runInContext(between('const localClock=', 'let gPromise='),ctx);
vm.runInContext(between('function localParts(', 'function homeForecastContext('),ctx);
function hours(start,count){return Array.from({length:count},(_,i)=>({time:Date.parse(start)+i*3600000,temp:i,rain:1,lead:i+1,wind:0,gust:0,cloud:0,pressure:1020}));}
// On a local day the ending rain interval belongs to the preceding day.
let rows=hours('2026-10-26T23:00:00Z',25),g=ctx.groups(rows).find(g=>g.date==='2026-10-27');
assert.equal(g.rs.length,24);assert.equal(g.rain,24);assert.equal(g.full,true);
// Spring and autumn DST days must have 23 and 25 observations respectively.
for(const [start,date,n]of [['2026-03-28T23:00:00Z','2026-03-29',23],['2026-10-24T22:00:00Z','2026-10-25',25]]){
 const g=ctx.groups(hours(start,n+1)).find(g=>g.date===date);assert.equal(g.rs.length,n);assert.equal(g.full,true);assert.equal(g.rain,n);
}
assert.equal(ctx.groups(hours('2026-10-27T12:00:00Z',10))[0].full,false);
vm.runInContext(between('function rainWindow(', 'function paintCurrentStrip('),ctx);
const period=ctx.rainWindow(Date.parse('2026-10-27T05:00:00Z')/1000);
assert.equal(new Date(period.start*1000).toISOString(),'2026-10-26T06:00:00.000Z');
assert.equal(new Date(period.end*1000).toISOString(),'2026-10-27T06:00:00.000Z');
// Solar strip survives removal of the old station node by the iframe plugin.
let sunBox={};ctx.$=id=>id==='#overview-sun'?sunBox:null;
ctx.sunTime=t=>new Date(t*1000).toISOString().slice(11,16);
vm.runInContext('let liveData={sun:{rise:1793078017,set:1793114398}};',ctx);
ctx.paintOverviewSun();assert.match(sunBox.innerHTML,/Wschód · czas polski/);assert.doesNotMatch(sunBox.innerHTML,/>—</);
assert.match(source,/#mk-current-strip/);assert.match(source,/rainConsistent/);
console.log('Local days: 24/23/25 hours; precipitation intervals; partial days; solar strip independent of iframe: OK');
