const {chromium}=require('playwright');
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',args:['--no-sandbox']});
 for(const station of ['turaszowka','lesniowka']){
  const page=await browser.newPage({viewport:{width:1366,height:1000},reducedMotion:'reduce'}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('https://meteo-krosno.pl/stacja-'+station+'/?verify=instruments',{waitUntil:'domcontentloaded',timeout:60000});
  await page.locator('#station-windsock').waitFor();
  await page.waitForFunction(()=>{const n=document.getElementById('rose-note');return n&&!n.textContent.includes('Odczytywanie');},null,{timeout:60000});
  const note=await page.locator('#rose-note').innerText();if(note.includes('Nie udało'))throw Error(note);
  if(await page.locator('#sock-fabric path').count()!==5)throw Error('Missing windsock fabric');
  if(await page.locator('#rain-cup-ticks text').count()!==6)throw Error('Missing rainfall scale');
  await page.locator('[data-rose-mode="period"]').click();
  await page.waitForFunction(()=>document.getElementById('rose-period-label').textContent.startsWith('Wybrany okres'));
  for(const width of [1366,390]){
   await page.setViewportSize({width,height:1000});
   await page.waitForTimeout(300);
   if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2))throw Error('Horizontal overflow '+station+' '+width);
   await page.screenshot({path:'/tmp/'+station+'-'+width+'.png',fullPage:true});
  }
  if(errors.length)throw Error(errors.join('\n'));
  console.log('PASS browser: '+station+' desktop/mobile, instruments, archive, period selector, no JS errors');await page.close();
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
