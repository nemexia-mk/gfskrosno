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
// The astronomy tile is independent from the iframe plugin and uses local days.
vm.runInContext(fs.readFileSync('site/vendor/astronomy-2.1.19.min.js','utf8'),ctx);
let noMoonEvent=false;
for(let day=1;day<=31;day++){
 const date='2026-10-'+String(day).padStart(2,'0'),data=ctx.mkAstronomyForDay(ctx.Astronomy,date);
 for(const k of ['sunrise','sunset','moonrise','moonset'])if(data[k]!==null)assert.equal(ctx.localParts(data[k]).date,date);
 if(data.moonrise===null||data.moonset===null)noMoonEvent=true;
 assert(data.sunset>data.sunrise);
}
assert(noMoonEvent,'A lunar event absent from a calendar day must remain absent');
const oct28=ctx.mkAstronomyForDay(ctx.Astronomy,'2026-10-28');
assert.equal(ctx.localParts(oct28.sunrise).hour,6);assert.equal(ctx.localParts(oct28.sunset).hour,16);
let sunBox={},duration={};ctx.$=id=>id==='#mk-astro-events'?sunBox:id==='#mk-day-duration'?duration:null;
vm.runInContext("let today='2026-10-28';mkAstroData=mkAstronomyForDay(Astronomy,today);",ctx);
ctx.paintOverviewSun(false);assert.match(sunBox.innerHTML,/Wschód księżyca/);assert.match(sunBox.innerHTML,/Zachód słońca/);assert.match(duration.textContent,/Długość dnia/);
vm.runInContext(between('function mkHomeRain(', 'function paintCurrentStrip('),ctx);
assert.equal(ctx.mkHomeRain(0),'brak opadu');assert.match(ctx.mkHomeRain(null),/Brak danych/);assert.equal(ctx.mkHomeRain(.04),'Opad <0,1 mm');
assert.equal(ctx.mkUpcomingHours(hours('2026-10-28T08:00:00Z',8),Date.parse('2026-10-28T09:20:00Z'))[0].time,Date.parse('2026-10-28T10:00:00Z'));
assert.equal(ctx.mkUpcomingHours(hours('2026-10-28T08:00:00Z',8),Date.parse('2026-10-28T09:20:00Z')).length,4);
assert.match(source,/#mk-current-strip/);assert.match(source,/rainConsistent/);
console.log('Local days 23/24/25h, rain intervals, Sun/Moon calendar-day boundaries, absent lunar events and upcoming hours: OK');
