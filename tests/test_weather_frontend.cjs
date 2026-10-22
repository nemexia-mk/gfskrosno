const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('site/app.js','utf8');
const valid=v=>typeof v==='number'&&Number.isFinite(v);
const run=Date.parse('2026-10-22T06:00:00Z');
const extract=(a,b)=>source.slice(source.indexOf(a),source.indexOf(b,source.indexOf(a)));
const ctx={valid,asset:'https://meteo-krosno.pl/wp-content/themes/meteo-krosno-modern/',AbortSignal,
 fetch:async()=>({ok:true,json:async()=>ctx.data})};
vm.createContext(ctx);vm.runInContext(extract('async function enrichGfs(', 'const stormColors='),ctx);
(async()=>{
 const row={date:'2026-10-22 06:00:00',lead:0,type:'Brak'};
 ctx.data={schema:1,run:'2026-10-22T06:00:00Z',rows:[{...row,humidity:85,cin:0,stormLevel:3,type:'Śnieg',solar:null}]};
 const g={run,rows:[{...row}]};await ctx.enrichGfs(g);
 assert.equal(g.extraAvailable,true);assert.equal(g.rows[0].humidity,85);assert.equal(g.rows[0].cin,0);
 assert.equal(g.rows[0].type,'Śnieg');assert.equal(g.rows[0].solar,null);
 ctx.data.run='2026-10-22T12:00:00Z';const old={run,rows:[{...row}]};await ctx.enrichGfs(old);
 assert.equal(old.extraAvailable,false);assert.equal(old.rows[0].type,'Brak');
 ctx.data.run='2026-10-22T06:00:00Z';ctx.data.rows[0].lead=6;
 const bad={run,rows:[{...row}]};await ctx.enrichGfs(bad);assert.equal(bad.extraAvailable,false);
 ctx.fetch=async()=>({ok:false});const missing={run,rows:[{...row}]};await ctx.enrichGfs(missing);
 assert.equal(missing.extraAvailable,false);
 vm.runInContext(extract('function mgPrecip(', 'function mgSvg('),ctx);
 assert.equal(ctx.mgPrecip({type:'Ziarna lodowe'}).color,'#a855f7');
 assert.equal(ctx.mgPrecip({type:'Deszcz marznący'}).color,'#ec4899');
 assert.equal(ctx.mgPrecip({type:'Śnieg'}).color,'#2384e8');
 assert.match(source,/data-storm-level/);assert.match(source,/Potencjał burzowy/);
 console.log('Frontend: poprawne łączenie runów, terminów, zer, braków i kolorów opadu.');
})().catch(e=>{console.error(e);process.exitCode=1});
