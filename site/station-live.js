'use strict';
const $=id=>document.getElementById(id), station=document.body.dataset.station;
const nf=new Intl.NumberFormat('pl-PL',{maximumFractionDigits:1,minimumFractionDigits:1});
const fmt=(v,d=1)=>v==null?'—':new Intl.NumberFormat('pl-PL',{maximumFractionDigits:d,minimumFractionDigits:d}).format(Math.abs(v)<0.5*10**(-d)?0:v);
const time=t=>t==null?'—':new Intl.DateTimeFormat('pl-PL',{timeZone:'Europe/Warsaw',hour:'2-digit',minute:'2-digit'}).format(new Date(t*1000));
const date=t=>new Intl.DateTimeFormat('pl-PL',{timeZone:'Europe/Warsaw',day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}).format(new Date(t*1000));
const val=(id,v,suffix='',d=1)=>{$(id).textContent=fmt(v,d)+(v==null?'':suffix);};

const utcDate=t=>new Intl.DateTimeFormat('pl-PL',{timeZone:'UTC',day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}).format(new Date(t*1000));
function fullHourTicks(start,end,target=6){
 const desired=(end-start)/3600/target;
 const step=[1,2,3,4,6,12,24,48,72,168].find(h=>h>=desired)||168;
 const hourFormatter=new Intl.DateTimeFormat('en-GB',{timeZone:'Europe/Warsaw',hour:'2-digit',hourCycle:'h23'});
 const dayFormatter=new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Warsaw',year:'numeric',month:'2-digit',day:'2-digit'});
 const result=[];
 for(let t=Math.ceil(start/3600)*3600;t<=end;t+=3600){
  const h=Number(hourFormatter.format(new Date(t*1000)));
  if(h%Math.min(step,24)!==0)continue;
  if(step>24){const parts=dayFormatter.formatToParts(new Date(t*1000));const get=k=>Number(parts.find(p=>p.type===k).value);const day=Math.floor(Date.UTC(get('year'),get('month')-1,get('day'))/86400000);if(day%Math.round(step/24)!==0)continue;}
  result.push(t);
 }
 return result;
}

let range='24h',field='temp',payload=null,busy=false,controller=null,requestVersion=0;
function update(d){
 if(payload&&view&&view[0]===payload.start&&view[1]===payload.end)view=null;payload=d;const per=d.period??{};for(const [id,key,suffix]of [['period-min','min',' °C'],['period-max','max',' °C'],['period-mean','mean',' °C'],['period-coverage','coverage','%']])val(id,per[key],suffix);val('period-amplitude',per.min==null||per.max==null?null:per.max-per.min,' °C');
 const l=d.latest,s=d.stats;const rainLabel=document.querySelector('.rain-day-label');if(rainLabel)rainLabel.textContent='Doba opadowa: 07:00 zimą / 08:00 latem czasu polskiego (06:00 UTC) do następnego dnia';const status={live:'Pomiar aktualny',delayed:'Dane opóźnione',offline:'Brak nowych danych',missing:'Brak pomiarów',clock:'Sprawdź czas stacji'};
 $('status').textContent=status[d.status]||'Brak danych';$('status').className='status '+d.status;
 $('updated').textContent=l?'Ostatni pomiar: '+date(l.ts):'Oczekiwanie na pierwszy odczyt';
 const notes=[...d.warnings];if(d.status==='offline')notes.unshift('Wyświetlamy ostatni dostępny pomiar. Stacja nie przesłała nowych danych od '+Math.max(0,Math.round(d.age/60))+' min.');
 if(d.status==='clock')notes.unshift('Czas pomiaru wyprzedza zegar serwera. Sprawdź ustawienia urządzenia.');
 $('notice').hidden=!notes.length;$('notice').textContent=notes.join(' ');
 val('temperature',l?.temp);val('humidity',l?.humidity,'',0);val('pressure',l?.pressure);val('dew',l?.dew,' °C');val('rain',s.rain);val('rainrate',l?.rainrate,' mm/h');
 val('tmin',s.min?.value,' °C');val('tmax',s.max?.value,' °C');val('tmean',s.mean,' °C');$('tmin-time').textContent=s.min?'godz. '+time(s.min.ts):'Brak historii dzisiaj';$('tmax-time').textContent=s.max?'godz. '+time(s.max.ts):'Brak historii dzisiaj';
 val('wind-card-speed',l?.wind);val('wind-card-gust',l?.gust,' km/h');val('gust',s.gust?.value);$('gust-time').textContent=s.gust?'godz. '+time(s.gust.ts)+' · dzisiaj':(l?.maxdailygust!=null?'Licznik stacji: '+fmt(l.maxdailygust)+' km/h · bez godziny':'Brak porywów w dzisiejszym archiwum');
 val('wind-card-speed',l?.wind);val('wind-card-gust',l?.gust,' km/h');$('wind-state').textContent=l?.wind==null?'Brak danych':(d.status==='live'?'Pomiar aktualny':'Ostatni odczyt');
 const dirs=['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];const dir=l?.direction;
 $('wind-card-direction').textContent=l?.wind===0?'Cisza · kierunek nieokreślony':dir==null?'Kierunek: brak danych':dirs[Math.round(((dir%360)+360)%360/22.5)%16]+' · '+fmt(dir,0)+'°';$('needle').style.opacity=dir==null||l?.wind===0?'0':'1';$('needle').style.transform='translate(-50%,-50%) rotate('+(dir??0)+'deg)';
 $('humidity-meter').style.width=Math.max(0,Math.min(100,l?.humidity??0))+'%';$('coverage-meter').style.width=s.coverage+'%';val('coverage',s.coverage);val('samples',s.samples,'',0);$('first').textContent=time(s.first);$('last').textContent=time(s.last);
 $('thermal').textContent=l?.temp==null?'Brak temperatury w paczce':(l.temp<0?'Temperatura poniżej zera':l.temp<10?'Chłodne powietrze':l.temp<20?'Umiarkowana temperatura':l.temp<30?'Ciepłe powietrze':'Wysoka temperatura');
 // Porownanie rzeczywistych ostatnich punktow z okolica godziny wczesniej.
 const last=d.chart.filter(p=>p.temp!=null).at(-1);const prev=last?d.chart.filter(p=>p.temp!=null&&p.ts<=last.ts-3600).at(-1):null;
 $('trend').textContent=last&&prev&&last.ts-prev.ts<5400&&d.serverTime-last.ts<600?(last.temp-prev.temp>=0?'↗ +':'↘ ')+fmt(last.temp-prev.temp)+'° / ~1 h':'Bieżący pomiar';
 const plast=d.chart.filter(p=>p.pressure!=null).at(-1),pfirst=plast?d.chart.filter(p=>p.pressure!=null&&p.ts<=plast.ts-10800).at(-1):null;
 $('pressure-trend').textContent=plast&&pfirst&&plast.ts-pfirst.ts<12600&&d.serverTime-plast.ts<600?'Zmiana ~3 h: '+(plast.pressure-pfirst.pressure>=0?'+':'')+fmt(plast.pressure-pfirst.pressure)+' hPa':'Brak danych do trendu 3 h';
 if($('ground-now')){val('ground-now',l?.sensor_tf_ch3);val('ground-min',s.groundMin?.value,' °C');val('ground-max',s.groundMax?.value,' °C');$('ground-min-time').textContent=s.groundMin?'godz. '+time(s.groundMin.ts):'Brak pomiarów';$('ground-max-time').textContent=s.groundMax?'godz. '+time(s.groundMax.ts):'Brak pomiarów';}
 val('solar-now',l?.solar,'',0);val('uv-now',l?.uv,'',1);val('solar-max',s.solarMax?.value,' W/m²',0);$('solar-max-time').textContent=s.solarMax?'· godz. '+time(s.solarMax.ts):'';val('rain-yesterday',s.rainYesterday);$('rain-yesterday-period').textContent=s.rainYesterdayStart?date(s.rainYesterdayStart)+' – '+date(s.rainYesterdayEnd)+' · czas polski (06:00–06:00 UTC)':'06:00–06:00 UTC';$('rain-yesterday-status').textContent=s.rainYesterday==null?'Brak pomiarów poprzedniej doby':s.rainYesterdayComplete?'Poprzednia zakończona doba opadowa':'Dane z części poprzedniej doby';
 $('sensors').replaceChildren();for(const sensor of l?.sensors??[]){const row=document.createElement('div');row.className='sensor-row';const label=document.createElement('span');label.textContent=sensor.label;const key=document.createElement('small');key.textContent=sensor.key;label.append(key);const value=document.createElement('strong');value.textContent=fmt(sensor.value)+(sensor.value==null?'':' °C');row.append(label,value);$('sensors').append(row);}
 if(!l)$('sensors').textContent='Oczekiwanie na dane czujników.';
 updateMonthly(d.monthly);drawChart();drawSpark();updateStationInstruments(d);
}
function updateMonthly(m){
 if(!m)return;$('month-label').textContent=m.label;val('month-temp',m.tempMean,' °C');val('month-rain',m.rain,' mm');val('month-frost',m.frostDays,' dni',0);val('month-sun-value',m.sunshine,' h');$('month-sun-norm').textContent='Norma: '+fmt(m.sunshineNorm)+' h';$('month-sun-percent').textContent=m.sunshinePercent==null?'Brak danych z tego miesiąca':fmt(m.sunshinePercent)+'% normy miesięcznej';$('month-sun-bar').style.width=Math.max(0,Math.min(100,m.sunshinePercent??0))+'%';$('month-sun-source').textContent='Krosno IMGW'+(m.sunshineThrough?' · dane do '+m.sunshineThrough.split('-').reverse().join('.')+' · '+m.sunshineDays+' dni':' · oczekiwanie na dane dobowe');
 $('month-temp-norm').textContent='Norma całego miesiąca: '+fmt(m.tempNorm)+' °C';
 $('month-temp-anomaly').textContent=m.tempAnomaly==null?'Brak pomiarów':(m.tempAnomaly>0?'+':'')+fmt(m.tempAnomaly)+' °C względem normy';
 const delta=m.tempAnomaly??0,bar=$('month-temp-bar');bar.style.left=(delta<0?50-Math.min(48,Math.abs(delta)*5):50)+'%';bar.style.width=Math.min(48,Math.abs(delta)*5)+'%';bar.style.background=delta<0?'#71bfff':'#ffb178';
 $('month-rain-percent').textContent=m.rainPercent==null?'Brak pomiarów':fmt(m.rainPercent)+'% normy miesięcznej';$('month-rain-norm').textContent='Norma: '+fmt(m.rainNorm)+' mm';$('month-rain-bar').style.width=Math.max(0,Math.min(100,m.rainPercent??0))+'%';
 $('month-frost-norm').textContent='Norma: '+fmt(m.frostNorm)+' dni';$('month-frost-percent').textContent=m.frostDays==null?'Brak pomiarów':m.frostNorm>0?fmt(m.frostDays/m.frostNorm*100)+'% normy · dostępny okres':'W dostępnych pomiarach';$('month-frost-bar').style.width=(m.frostNorm>0?Math.min(100,(m.frostDays??0)/m.frostNorm*100):m.frostDays>0?100:0)+'%';
 $('month-coverage').textContent=m.first?'Pomiary od '+date(m.first)+'. Dostępność od początku miesiąca: '+fmt(m.coverage)+'%. Średnia temperatury i liczba dni z przymrozkiem dotyczą dostępnych pomiarów; suma opadu pochodzi z licznika miesięcznego stacji.':'Oczekiwanie na pomiary z tego miesiąca.';
}
const meta={temp:{label:'Temperatura',unit:'°C',color:'#76edbd'},humidity:{label:'Wilgotność',unit:'%',color:'#65bfff'},pressure:{label:'Ciśnienie',unit:'hPa',color:'#c5afff'},wind:{label:'Wiatr',unit:'km/h',color:'#ffc48b'},rain:{label:'Opad dobowy',unit:'mm',color:'#65bfff'},rainrate:{label:'Natężenie opadu',unit:'mm/h',color:'#65bfff'},solar:{label:'Promieniowanie',unit:'W/m²',color:'#ffcf85'},uv:{label:'UV',unit:'indeks',color:'#c5afff'}};
const ns='http://www.w3.org/2000/svg';function svgEl(name,attrs,text){const n=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs||{}))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n;}
function paths(points,key,x,y,step){let str='',last=null;for(const p of points){if(p[key]==null){last=null;continue;}str+=(last===null||p.ts-last>step*1.5?'M':'L')+x(p.ts).toFixed(2)+','+y(p[key]).toFixed(2)+' ';last=p.ts;}return str;}
function drawSpark(){const svg=$('spark');svg.replaceChildren();if(!payload)return;const ps=payload.chart.filter(p=>p.temp!=null);if(ps.length<2)return;const lo=Math.min(...ps.map(p=>p.temp)),hi=Math.max(...ps.map(p=>p.temp));svg.append(svgEl('path',{d:paths(payload.chart,'temp',t=>(t-payload.start)/(payload.end-payload.start||1)*600,v=>58-(v-lo)/(hi-lo||1)*50,payload.step),fill:'none',stroke:'#76edbd','stroke-width':2}));}
let view=null,hoverIndex=-1;
function visiblePoints(){if(!payload)return [];return payload.chart.filter(p=>p.ts>=view[0]&&p.ts<=view[1]);}
function renderTable(){const body=$('history-rows');body.replaceChildren();for(const p of visiblePoints().slice(-100).reverse()){const tr=document.createElement('tr');for(const v of [date(p.ts),...['temp','humidity','pressure','wind','gust','rain'].map(k=>fmt(p[k]))]){const td=document.createElement('td');td.textContent=v;tr.append(td);}body.append(tr);}}
function drawChart(){
 const svg=$('chart');svg.replaceChildren();$('tooltip').hidden=true;if(!payload)return;
 if(!view)view=[payload.start,payload.end];
 const {unit,color}=meta[field],ps=visiblePoints();$('chart-unit').textContent=unit;
 let series=[{key:field,label:meta[field].label,color}];
 if(field==='temp'&&$('show-dew').checked)series.push({key:'dew',label:'Punkt rosy',color:'#cca4ff',dash:'3 5'});
 if(field==='wind')series.push({key:'gust',label:'Poryw',color:'#ff987e',dash:'6 4'});
 if(field==='temp'&&$('show-sensors').checked){for(const [i,s]of (payload.latest?.sensors??[]).entries())series.push({key:'sensor_'+s.key,label:s.label,color:['#ffcf85','#79caff','#f698c6'][i%3],dash:'7 4'});}
 $('chart-legend').replaceChildren();for(const s of series){const item=document.createElement('span');const swatch=document.createElement('i');swatch.style.background=s.color;item.append(swatch,document.createTextNode(s.label));$('chart-legend').append(item);}
 const vals=ps.flatMap(p=>series.map(s=>p[s.key]).concat(field==='temp'&&$('show-band').checked?[p.low,p.high]:[])).filter(v=>Number.isFinite(v));
 $('chart-empty').hidden=!!vals.length;$('chart-empty').textContent=ps.length?'Brak tego parametru w zapisanych pomiarach.':'Brak pomiarów w wybranym okresie.';
 const W=Math.max(480,Math.round(svg.getBoundingClientRect().width)),H=340,L=55,R=W-20,T=28,B=285;
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('preserveAspectRatio','none');
 let lo=vals.length?Math.min(...vals):0,hi=vals.length?Math.max(...vals):1;const pad=Math.max((hi-lo)*.15,field==='pressure'?1:field==='temp'?.5:1);lo-=pad;hi+=pad;
 if(['humidity','wind','rain','rainrate','solar','uv'].includes(field))lo=Math.max(0,lo);if(field==='humidity')hi=Math.min(100,hi);if(hi<=lo)hi=lo+1;
 const x=t=>L+(t-view[0])/(view[1]-view[0]||1)*(R-L),y=v=>B-(v-lo)/(hi-lo)*(B-T);
 const defs=svgEl('defs'),grad=svgEl('linearGradient',{id:'area-fill',x1:0,x2:0,y1:0,y2:1});grad.append(svgEl('stop',{offset:'0%','stop-color':color,'stop-opacity':.2}),svgEl('stop',{offset:'100%','stop-color':color,'stop-opacity':0}));defs.append(grad);svg.append(defs);
 for(let i=0;i<=5;i++){const v=lo+(hi-lo)*i/5,yy=y(v);svg.append(svgEl('line',{x1:L,x2:R,y1:yy,y2:yy,stroke:'#29434e','stroke-dasharray':'3 6'}),svgEl('text',{x:L-10,y:yy+4,fill:'#9eb6c2','font-size':11,'text-anchor':'end'},fmt(v,['humidity','solar'].includes(field)?0:1)));}
 const ticks=fullHourTicks(view[0],view[1],W<650?3:6);for(const t of ticks){const xx=x(t);svg.append(svgEl('line',{x1:xx,x2:xx,y1:T,y2:B,stroke:'#203a46','stroke-dasharray':'3 6'}),svgEl('text',{x:xx,y:314,fill:'#9eb6c2','font-size':11,'text-anchor':xx<L+35?'start':xx>R-35?'end':'middle'},view[1]-view[0]>90000?date(t):time(t)));}
 // Ciagle fragmenty; nigdy nie laczymy przerw w odbiorze.
 function segments(key){const groups=[];let g=[];for(const p of ps){if(p[key]==null||(g.length&&p.ts-g.at(-1).ts>payload.step*1.5)){if(g.length)groups.push(g);g=[];}if(p[key]!=null)g.push(p);}if(g.length)groups.push(g);return groups;}
 if(field==='temp'&&$('show-band').checked){for(const g of segments('low')){const upper=g.filter(p=>p.high!=null);if(!upper.length)continue;const d='M'+upper.map(p=>x(p.ts)+','+y(p.high)).join(' L')+' L'+upper.slice().reverse().map(p=>x(p.ts)+','+y(p.low)).join(' L')+' Z';svg.append(svgEl('path',{d,fill:color,opacity:.14}));}}
 if(field!=='temp'&&field!=='rainrate'){for(const g of segments(field)){if(g.length<2)continue;svg.append(svgEl('path',{d:paths(g,field,x,y,payload.step)+'L'+x(g.at(-1).ts)+','+B+' L'+x(g[0].ts)+','+B+' Z',fill:'url(#area-fill)'}));}}
 for(const s of series){if(field==='rainrate'&&s.key==='rainrate'){const width=Math.max(1,Math.min(35,payload.step/(view[1]-view[0])*(R-L)*.75));for(const p of ps)if(p.rainrate!=null)svg.append(svgEl('rect',{x:x(p.ts)-width/2,y:y(p.rainrate),width,height:Math.max(1,B-y(p.rainrate)),rx:2,fill:s.color,opacity:.8}));}else{svg.append(svgEl('path',{d:paths(ps,s.key,x,y,payload.step),fill:'none',stroke:s.color,'stroke-width':s.key===field?2.8:1.8,'stroke-dasharray':s.dash||'','stroke-linejoin':'round','stroke-linecap':'round','vector-effect':'non-scaling-stroke'}));for(const p of ps)if(p[s.key]!=null&&ps.length<600)svg.append(svgEl('circle',{cx:x(p.ts),cy:y(p[s.key]),r:ps.length<3?3:1.3,fill:s.color}));}}
 const valid=ps.filter(p=>p[field]!=null);if(valid.length){const min=valid.reduce((a,b)=>b[field]<a[field]?b:a),max=valid.reduce((a,b)=>b[field]>a[field]?b:a);for(const [p,label]of [[min,'min'],[max,'max']]){svg.append(svgEl('circle',{cx:x(p.ts),cy:y(p[field]),r:4,fill:'#102530',stroke:color,'stroke-width':2}));if(valid.length>1)svg.append(svgEl('text',{x:Math.max(L+5,Math.min(R-70,x(p.ts)+8)),y:Math.max(T+12,y(p[field])-12),fill:color,'font-size':11},label+' '+fmt(p[field])));}
 $('chart-summary').textContent=`Widoczny zakres: ${date(view[0])} – ${date(view[1])} · min ${fmt(min[field])} ${unit} · max ${fmt(max[field])} ${unit} · ${valid.length} przedziałów z danymi`;
 }else $('chart-summary').textContent='Brak wartości dla tego parametru w widocznym zakresie.';
 $('chart-note').textContent=(field==='temp'?'Linia: średnia w przedziale. Cieniowanie: rzeczywiste minimum i maksimum w przedziale. ':field==='rainrate'?'Słupki: średnie natężenie opadu w przedziale (mm/h), nie suma opadu. ':field==='rain'?'Licznik dobowy stacji; spadek oznacza jego zerowanie. ':'')+`Rozdzielczość ${payload.step/60} min. Przeciągnij po wykresie, aby przybliżyć. Strzałki klawiatury przeglądają odczyty. Luki nie są uzupełniane.`;
 const hover=svgEl('line',{x1:0,x2:0,y1:T,y2:B,stroke:'#c0e6d7','stroke-dasharray':'3 3',visibility:'hidden'}),selection=svgEl('rect',{x:0,y:T,width:0,height:B-T,fill:'#76edbd',opacity:.13});svg.append(selection,hover);
 function showPoint(index){const p=ps[index];if(!p)return;hoverIndex=index;hover.setAttribute('x1',x(p.ts));hover.setAttribute('x2',x(p.ts));hover.setAttribute('visibility','visible');const tip=$('tooltip');tip.replaceChildren();const title=document.createElement('strong');title.textContent=date(p.ts);tip.append(title);for(const s of series){const row=document.createElement('div');row.textContent=s.label+': '+fmt(p[s.key])+' '+unit;row.style.color=s.color;tip.append(row);}if(field==='temp'&&p.low!=null){const row=document.createElement('div');row.textContent='Tmin / Tmax: '+fmt(p.low)+' / '+fmt(p.high)+' °C';tip.append(row);}tip.hidden=false;}
 let drag=null;const sx=e=>(e.clientX-svg.getBoundingClientRect().left)/svg.getBoundingClientRect().width*W;
 svg.onpointermove=e=>{const xx=Math.max(L,Math.min(R,sx(e)));if(drag!==null){selection.setAttribute('x',Math.min(xx,drag));selection.setAttribute('width',Math.abs(xx-drag));}const target=view[0]+(xx-L)/(R-L)*(view[1]-view[0]);let index=-1;ps.forEach((p,i)=>{if(index<0||Math.abs(p.ts-target)<Math.abs(ps[index].ts-target))index=i;});if(index>=0&&Math.abs(ps[index].ts-target)<=payload.step*1.5)showPoint(index);else{$('tooltip').hidden=true;hover.setAttribute('visibility','hidden');}};
 svg.onpointerdown=e=>{if(sx(e)<L||sx(e)>R)return;drag=sx(e);svg.setPointerCapture(e.pointerId);};
 svg.onpointerup=e=>{if(drag===null)return;const xx=Math.max(L,Math.min(R,sx(e))),left=Math.min(drag,xx),right=Math.max(drag,xx);if(right-left>25){const a=view[0]+(left-L)/(R-L)*(view[1]-view[0]),b=view[0]+(right-L)/(R-L)*(view[1]-view[0]);if(b-a>=payload.step*2){view=[a,b];drawChart();}}drag=null;selection.setAttribute('width',0);};
 svg.onpointercancel=()=>{drag=null;selection.setAttribute('width',0);};svg.onpointerleave=()=>{if(drag===null){$('tooltip').hidden=true;hover.setAttribute('visibility','hidden');}};
 svg.onkeydown=e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();showPoint(Math.max(0,Math.min(ps.length-1,hoverIndex+(e.key==='ArrowRight'?1:-1))));}if(e.key==='Escape'){$('tooltip').hidden=true;}};
 renderTable();queueStationRose();
}

let csvRefreshFlight=false,lastCsvRefresh=0;
async function refreshCsvInBackground(){
 if(csvRefreshFlight||Date.now()-lastCsvRefresh<45000)return;csvRefreshFlight=true;lastCsvRefresh=Date.now();
 try{
  const endpoint=new URL(document.body.dataset.refreshApi);endpoint.search=new URLSearchParams({station}).toString();
  const response=await fetch(endpoint.href,{cache:'no-store'});if(!response.ok)return;
  const result=await response.json();if(result.refreshed)load();
 }catch(e){}finally{csvRefreshFlight=false;}
}
function loadingStage(stage,detail){
 $('loading-progress').value=stage;$('loading-step').textContent='Etap '+Math.min(3,stage+1)+' z 3 · '+['Pobieranie pomiarów','Odczytywanie danych','Rysowanie wykresów','Gotowe'][stage];$('loading-detail').textContent=detail;
}
async function load(force=false){
 if(busy&&!force)return;if(controller)controller.abort();controller=new AbortController();const v=++requestVersion;busy=true;$('refresh').disabled=true;
 const showLoading=!payload||force;if(showLoading){$('data-loading').hidden=false;loadingStage(0,'Oczekiwanie na odpowiedź stacji…');}
 const activeController=controller,timer=setTimeout(()=>activeController.abort(),45000);
 try{
  let response=await fetch(document.body.dataset.api+(document.body.dataset.api.includes('?')?'&':'?')+'station='+encodeURIComponent(station)+'&range='+range+'&date='+encodeURIComponent($('archive-date').value)+'&resolution='+$('resolution').value,{cache:'no-store',signal:activeController.signal});
  if(response.status===503&&document.body.dataset.fallbackApi){const fallback=new URL(document.body.dataset.fallbackApi);fallback.search=new URLSearchParams({station,range,date:$('archive-date').value,resolution:$('resolution').value}).toString();response=await fetch(fallback.href,{cache:'no-store',signal:activeController.signal});}
  if(v!==requestVersion)return;if(!response.ok)throw Error('HTTP '+response.status);
  if(showLoading)loadingStage(1,'Odebrano odpowiedź · odczytywanie pomiarów');
  const d=await response.json();if(v!==requestVersion)return;
  if(showLoading){loadingStage(2,'Przygotowywanie widoku stacji');await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));}
  if(v!==requestVersion)return;update(d);loadStationRose();if(showLoading)loadingStage(3,'Pomiary i wykresy są gotowe');
 }catch(e){if(v!==requestVersion)return;stopStationSock();$('status').className='status offline';$('status').textContent='Błąd połączenia';$('notice').hidden=false;$('notice').textContent='Nie udało się odczytać pomiarów. '+(payload?'Pokazujemy dane z poprzedniego odświeżenia. ':'')+'Spróbujemy ponownie automatycznie.';}
 finally{clearTimeout(timer);if(v===requestVersion){busy=false;$('refresh').disabled=false;$('data-loading').hidden=true;if(payload?.refreshRequired)refreshCsvInBackground();}}
}
document.querySelectorAll('[data-range]').forEach(b=>b.addEventListener('click',()=>{range=b.dataset.range;view=null;document.querySelectorAll('[data-range]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));load(true);}));
document.querySelectorAll('[data-field]').forEach(b=>b.addEventListener('click',()=>{field=b.dataset.field;document.querySelectorAll('[data-field]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));drawChart();}));
$('refresh').addEventListener('click',()=>load(true));
const today=new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Warsaw',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
$('archive-date').value=today;$('archive-date').max=today;
$('archive-date').addEventListener('change',()=>{if(!$('archive-date').value)return;range='day';view=null;document.querySelectorAll('[data-range]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.range==='day')));load(true);});
$('resolution').addEventListener('change',()=>{view=null;load(true);});
for(const id of ['show-dew','show-band','show-sensors'])$(id).addEventListener('change',drawChart);
function zoom(factor,shift=0){if(!payload||!view)return;const full=payload.end-payload.start,width=Math.min(full,Math.max(payload.step*3,(view[1]-view[0])*factor));let a=(view[0]+view[1])/2-width/2+shift*(view[1]-view[0]);a=Math.max(payload.start,Math.min(payload.end-width,a));view=[a,a+width];drawChart();}
$('zoom-in').addEventListener('click',()=>zoom(.5));$('zoom-out').addEventListener('click',()=>zoom(2));$('pan-left').addEventListener('click',()=>zoom(1,-.5));$('pan-right').addEventListener('click',()=>zoom(1,.5));$('zoom-reset').addEventListener('click',()=>{view=null;drawChart();});
function expandPanel(force){const panel=$('history'),open=force??!panel.classList.contains('expanded');panel.classList.toggle('expanded',open);$('expand-chart').setAttribute('aria-pressed',String(open));$('expand-chart').textContent=open?'⤡ Zamknij powiększenie':'⤢ Powiększ panel';document.body.classList.toggle('chart-open',open);drawChart();}
$('expand-chart').addEventListener('click',()=>expandPanel());document.addEventListener('keydown',e=>{if(e.key==='Escape')expandPanel(false);});
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawChart,150);});
function clock(){$('clock').textContent=new Intl.DateTimeFormat('pl-PL',{timeZone:'Europe/Warsaw',day:'numeric',month:'long',hour:'2-digit',minute:'2-digit'}).format(new Date());}clock();setInterval(clock,1000);load();setInterval(()=>{if(!document.hidden)load();},60000);document.addEventListener('visibilitychange',()=>{if(!document.hidden)load();});

// Dopasowanie wysokosci osadzonego dashboardu bez obcych domen.
function reportHeight(){if(window.parent!==window)window.parent.postMessage({type:"mk-station-height",station,height:Math.ceil(document.documentElement.scrollHeight)},window.location.origin);}
new ResizeObserver(reportHeight).observe(document.body);window.addEventListener("load",reportHeight);reportHeight();


// Dodatkowe instrumenty stacji. Jednostka wiatru: km/h; opad: licznik dobowy w mm.
const stationWindColors=['#65bfff','#76edbd','#f6d777','#ffac76','#ef91b1'];
let sockFrame=0,sockState={speed:null,fresh:false,ts:null},roseMode='today',roseTimer=0,roseAbort=null,roseSerial=0,roseSignature='',roseData=null;
const stationFinite=v=>typeof v==='number'&&Number.isFinite(v);

function stationSockShape(speed,phase=0,reduced=false){
 const known=stationFinite(speed)&&speed>=0,wind=known?Math.min(200,speed):0;
 const extension=1-Math.exp(-wind/14),amplitude=reduced?0:Math.min(12,wind*.32),length=70+86*extension,droop=82*(1-extension);
 const nodes=Array.from({length:6},(_,i)=>{const f=i/5;return {x:length*f,y:droop*f*f+amplitude*f*f*Math.sin(phase-i*.8),half:15-9*f};});
 return nodes.slice(0,-1).map((a,i)=>{const b=nodes[i+1];return 'M'+a.x+','+(a.y-a.half)+' L'+b.x+','+(b.y-b.half)+' L'+b.x+','+(b.y+b.half)+' L'+a.x+','+(a.y+a.half)+' Z';});
}
function stationRainScale(value){
 if(!stationFinite(value)||value<0)return {known:false,max:5,fraction:0};
 let max=[5,10,20,50,100,200,500,1000].find(n=>value<=n);
 if(!max)max=Math.ceil(value/1000)*1000;
 return {known:true,max,fraction:Math.min(1,value/max)};
}
function mountStationInstruments(){
 const wind=document.querySelector('.wind-panel');
 if(wind&&!$('station-windsock')){
  const row=document.createElement('div');row.className='station-wind-motion';
  row.innerHTML='<div class="windsock-visual"><svg id="station-windsock" viewBox="0 0 240 180" role="img" aria-label="Rękaw wiatru — oczekiwanie na pomiar"><defs><linearGradient id="sock-pole" x1="0" x2="1"><stop stop-color="#546e7a"/><stop offset="50%" stop-color="#c4d9dc"/><stop offset="100%" stop-color="#546e7a"/></linearGradient></defs><ellipse cx="53" cy="169" rx="43" ry="5" fill="#020e15" opacity=".4"/><path d="M35 156V24" stroke="#c5dade" stroke-width="5"/><path d="M35 36H47" stroke="#d5e6e8" stroke-width="3"/><path d="M35 145L15 168M35 145L57 168M35 145V172" fill="none" stroke="#9ab8c1" stroke-width="4" stroke-linecap="round"/><path d="M10 171H20M52 171H62M30 175H40" stroke="#d1e3e6" stroke-width="3"/><g transform="translate(47 36)"><g id="sock-fabric"></g><ellipse cx="0" cy="0" rx="3" ry="15" fill="none" stroke="#d9e7e6" stroke-width="2"/></g></svg><span id="sock-caption">Brak pomiaru wiatru</span></div><div class="wind-dominant"><span class="eyebrow">DOMINUJĄCY KIERUNEK DZISIAJ</span><strong id="dominant-wind-today">Oczekiwanie na dane</strong><small id="dominant-wind-note">Z zapisanych kierunków wiatru</small></div>';
  wind.querySelector('.wind-main-content').append(row.querySelector('.windsock-visual'));wind.append(row);
 }
 const rain=document.querySelector('.metric.rain');
 if(rain&&!$('station-rain-cup')){
  const row=document.createElement('div');row.className='station-rain-row';const text=document.createElement('div');text.className='station-rain-readout';
  const value=rain.querySelector('.metric-value'),note=rain.querySelector('p');if(value)text.append(value);if(note)text.append(note);row.append(text);
  const visual=document.createElement('div');visual.className='rain-cup-visual';
  visual.innerHTML='<svg id="station-rain-cup" viewBox="0 0 132 166" role="img" aria-label="Kubeczek opadów — oczekiwanie na dane"><defs><clipPath id="rain-cup-clip"><path d="M15 27H83L75 143H23Z"/></clipPath><linearGradient id="cup-water" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#72d8ff"/><stop offset="100%" stop-color="#278bc8"/></linearGradient></defs><g class="rain-drops" fill="#65bfff"><path d="M35 0q-7 11 0 11q7 0 0-11"/><path d="M60 -5q-6 10 0 10q6 0 0-10"/></g><path d="M15 27H83L75 143H23Z" fill="#65bfff0a"/><g clip-path="url(#rain-cup-clip)"><g id="rain-cup-water" class="rain-cup-water" style="transform:translateY(145px)"><rect x="12" y="0" width="75" height="145" fill="url(#cup-water)" opacity=".85"/><path class="cup-wave" d="M5 2Q18 -3 32 2T59 2T86 2T113 2" fill="none" stroke="#a6efff" stroke-width="2"/></g></g><path d="M15 27L23 143H75L83 27M10 27H88" fill="none" stroke="#94cce0" stroke-width="2.5" stroke-linejoin="round"/><g id="rain-cup-ticks"></g><text x="102" y="15" text-anchor="middle" fill="#a9cddd" font-size="10">mm</text></svg><small id="rain-cup-scale">Skala w mm</small>';
  row.append(visual);rain.append(row);
 }
 if(!$('wind-rose')){
  const section=document.createElement('section');section.id='wind-rose';section.className='panel station-wind-rose';
  section.innerHTML='<div class="section-heading"><div><p class="eyebrow">SKĄD WIAŁ WIATR</p><h2>Róża wiatrów</h2><p class="muted" id="rose-period-label">Dzisiaj · czas polski</p></div><div class="segmented" aria-label="Okres róży wiatrów"><button data-rose-mode="today" aria-pressed="true">Dzisiaj</button><button data-rose-mode="period" aria-pressed="false">Wybrany okres wykresu</button></div></div><div class="rose-layout"><div><svg id="wind-rose-svg" viewBox="0 0 420 420" role="img" aria-label="Róża wiatrów — oczekiwanie na dane"></svg><div id="rose-speed-legend" class="rose-speed-legend"></div></div><div class="rose-explanation"><div class="rose-stat"><span>Dominujący kierunek okresu</span><strong id="rose-dominant">—</strong></div><div class="rose-stat"><span>Cisza (poniżej 1,8 km/h)</span><strong id="rose-calm">—</strong></div><div class="rose-stat"><span>Czas z danymi o kierunku lub ciszy</span><strong id="rose-observed">—</strong></div><div class="rose-stat"><span>Pokrycie wybranego okresu</span><strong id="rose-coverage">—</strong></div><p id="rose-note" class="muted" role="status">Odczytywanie archiwum…</p><details class="data-table"><summary>Udziały kierunków</summary><div class="table-scroll"><table><thead><tr><th>Kierunek</th><th>Udział czasu</th></tr></thead><tbody id="rose-table"></tbody></table></div></details></div></div><p class="chart-note">Róża wskazuje, skąd wiał wiatr. Długość ramienia oznacza udział czasu, kolory — przedziały prędkości w km/h. Ciszę pokazujemy osobno.</p>';
  const details=document.querySelector('section.details');if(details)details.before(section);else document.querySelector('main').append(section);
  for(const b of section.querySelectorAll('[data-rose-mode]'))b.addEventListener('click',()=>{roseMode=b.dataset.roseMode;for(const x of section.querySelectorAll('[data-rose-mode]'))x.setAttribute('aria-pressed',String(x===b));roseSignature='';queueStationRose();});
  const nav=document.querySelector('.page-jumps');if(nav){const a=document.createElement('a');a.href='#wind-rose';a.textContent='Róża wiatrów';nav.append(a);}
  const labels=['1,8–5','5–10','10–20','20–30','≥30'];
  $('rose-speed-legend').innerHTML=labels.map((l,i)=>'<span><i style="background:'+stationWindColors[i]+'"></i>'+l+' km/h</span>').join('');
 }
 paintSock(0,true);paintRainCup(null,null);paintRose(null);
}
function paintSock(now=0,still=false){
 const fabric=$('sock-fabric');if(!fabric)return;
 const speed=sockState.speed,phase=now/1000*(1.5+Math.min(80,speed||0)*.2);
 fabric.replaceChildren();
 stationSockShape(speed,phase,still).forEach((d,i)=>fabric.append(svgEl('path',{d,fill:stationFinite(speed)?(i%2?'#f0f7f5':'#ff9d57'):(i%2?'#a7b5b8':'#5c727d'),stroke:'#223e49','stroke-width':.6})));
}
function stationReducedMotion(){return window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;}
function runStationSock(now){
 if(document.hidden||stationReducedMotion()||!sockState.fresh||(sockState.ts&&Date.now()/1000-sockState.ts>900)||!stationFinite(sockState.speed)||sockState.speed<=0){paintSock(0,true);sockFrame=0;return;}
 paintSock(now,false);sockFrame=requestAnimationFrame(runStationSock);
}
function startStationSock(){if(sockFrame)cancelAnimationFrame(sockFrame);sockFrame=0;runStationSock(performance.now());}
function stopStationSock(){sockState.fresh=false;startStationSock();if($('sock-caption'))$('sock-caption').textContent=stationFinite(sockState.speed)?'Ostatni pomiar · animacja wstrzymana':'Brak pomiaru wiatru';}
function paintRainCup(value,rate){
 const svg=$('station-rain-cup');if(!svg)return;const scale=stationRainScale(value);
 $('rain-cup-water').style.transform='translateY('+(145-scale.fraction*118)+'px)';$('rain-cup-water').style.opacity=scale.known?'1':'0';
 svg.classList.toggle('is-raining',scale.known&&stationFinite(rate)&&rate>0);svg.classList.toggle('has-water',scale.known&&value>0);
 svg.setAttribute('aria-label',scale.known?'Opad dzisiaj '+fmt(value)+' mm, podziałka do '+fmt(scale.max,0)+' mm':'Brak danych o opadzie');
 $('rain-cup-scale').textContent=scale.known?'Skala 0–'+fmt(scale.max,0)+' mm':'Brak danych';
 const ticks=$('rain-cup-ticks');ticks.replaceChildren();
 for(let i=0;i<=5;i++){const y=143-i*116/5;ticks.append(svgEl('path',{d:'M86 '+y+'h7',stroke:'#789aa9','stroke-width':1}),svgEl('text',{x:98,y:y+3,fill:'#b5cfdb','font-size':10},fmt(scale.max*i/5,scale.max<5?1:0)));}
}
function updateStationInstruments(d){
 const l=d.latest;sockState={speed:stationFinite(l?.wind)&&l.wind>=0?l.wind:null,fresh:d.status==='live',ts:l?.ts??null};
 const label=sockState.speed===null?'Brak pomiaru wiatru':!sockState.fresh?'Ostatni pomiar · animacja wstrzymana':sockState.speed===0?'Cisza · 0 km/h':fmt(sockState.speed)+' km/h';
 $('sock-caption').textContent=label;$('station-windsock').setAttribute('aria-label',label);startStationSock();paintRainCup(d.stats?.rain,l?.rainrate);
}
function rosePolar(radius,angle){const a=angle*Math.PI/180;return [210+radius*Math.sin(a),210-radius*Math.cos(a)];}
function roseWedge(inner,outer,angle){
 const a=rosePolar(outer,angle-7.5),b=rosePolar(outer,angle+7.5),c=rosePolar(inner,angle+7.5),d=rosePolar(inner,angle-7.5);
 return 'M'+a.join(',')+' A'+outer+','+outer+' 0 0 1 '+b.join(',')+' L'+c.join(',')+(inner>0?' A'+inner+','+inner+' 0 0 0 '+d.join(','):'')+' Z';
}
function paintRose(data){
 const svg=$('wind-rose-svg');if(!svg)return;svg.replaceChildren();
 if(!data||!data.validSeconds){svg.append(svgEl('circle',{cx:210,cy:210,r:155,fill:'#0b202b',stroke:'#294651','stroke-dasharray':'3 6'}),svgEl('text',{x:210,y:210,'text-anchor':'middle',fill:'#9eb6c2','font-size':13},data?'Brak danych o wietrze':'Oczekiwanie na archiwum'));svg.setAttribute('aria-label','Róża wiatrów — brak danych');return;}
 const max=Math.max(5,Math.ceil(Math.max(...data.sectors.map(s=>s.percent||0))/5)*5),radius=146;
 for(let i=1;i<=4;i++){const r=radius*i/4;svg.append(svgEl('circle',{cx:210,cy:210,r,fill:'none',stroke:'#31515e','stroke-dasharray':'3 5'}),svgEl('text',{x:216,y:210-r-4,fill:'#95afbd','font-size':10},fmt(max*i/4,1)+'%'));}
 data.sectors.forEach((sector,i)=>{const pos=rosePolar(174,i*22.5),edge=rosePolar(radius,i*22.5);svg.append(svgEl('path',{d:'M210 210L'+edge.join(','),stroke:'#29434e','stroke-width':.7}),svgEl('text',{x:pos[0],y:pos[1]+4,'text-anchor':'middle',fill:i%4===0?'#e0f1f5':'#96b5c2','font-size':i%4===0?13:10},sector.code));let cumulative=0;
  sector.binPercent.forEach((percent,j)=>{if(percent<=0)return;const inner=radius*cumulative/max;cumulative+=percent;const outer=radius*cumulative/max;const petal=svgEl('path',{d:roseWedge(inner,outer,i*22.5),fill:stationWindColors[j],stroke:'#102530','stroke-width':.65});petal.append(svgEl('title',{},sector.name+' ('+sector.code+') · '+fmt(percent,1)+'% czasu · '+['1,8–5','5–10','10–20','20–30','≥30'][j]+' km/h'));svg.append(petal);});
 });
 svg.append(svgEl('circle',{cx:210,cy:210,r:28,fill:'#102530',stroke:'#456775'}),svgEl('text',{x:210,y:208,'text-anchor':'middle',fill:'#e0f1f5','font-size':13},fmt(data.calmPercent,0)+'%'),svgEl('text',{x:210,y:222,'text-anchor':'middle',fill:'#9eb6c2','font-size':9},'cisza'));
 svg.setAttribute('aria-label','Róża wiatrów, dominujący '+(data.dominant?.name||'cisza')+', cisza '+fmt(data.calmPercent)+' procent');
}
function stationDominantText(data){return data?.dominant?data.dominant.name+' ('+data.dominant.code+')':data?.validSeconds>0?'Cisza — bez dominującego kierunku':'Brak danych o kierunku';}
function paintStationRoseData(d){
 roseData=d;const day=d.today,selected=roseMode==='today'?day:d.period;
 $('dominant-wind-today').textContent=stationDominantText(day);
 $('dominant-wind-note').textContent=day.validSeconds?fmt(day.coverage)+'% dzisiejszego okresu z danymi'+(day.dominant?' · '+fmt(day.dominant.percent)+'% czasu z kierunku '+day.dominant.code:''):'Brak zapisanych prędkości lub kierunków dzisiaj';
 $('rose-period-label').textContent=(roseMode==='today'?'Dzisiaj':'Wybrany okres')+' · '+date(selected.start)+' – '+date(selected.end);
 $('rose-dominant').textContent=stationDominantText(selected);$('rose-calm').textContent=selected.calmPercent===null?'—':fmt(selected.calmPercent)+'%';
 $('rose-observed').textContent=selected.validSeconds?fmt(selected.validSeconds/3600)+' h':'—';$('rose-coverage').textContent=fmt(selected.coverage)+'%';
 $('rose-note').textContent=selected.validSeconds?d.method+(selected.missingDirectionSeconds>0?' Pominięto '+fmt(selected.missingDirectionSeconds/60,0)+' min z prędkością, ale bez kierunku.':''):'Brak pomiarów wiatru w tym okresie. Róża pojawi się po zapisaniu prędkości i kierunku.';
 const body=$('rose-table');body.replaceChildren();for(const sector of selected.sectors){const tr=document.createElement('tr');for(const text of [sector.code+' · '+sector.name,sector.percent===null?'—':fmt(sector.percent)+'%']){const td=document.createElement('td');td.textContent=text;tr.append(td);}body.append(tr);}
 paintRose(selected);
}
function queueStationRose(){
 if(!payload||!$('wind-rose'))return;clearTimeout(roseTimer);roseTimer=setTimeout(loadStationRose,180);
}
function csvWindRose(start,end){
 const codes=['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
 const names=['północny','północno-północno-wschodni','północno-wschodni','wschodnio-północno-wschodni','wschodni','wschodnio-południowo-wschodni','południowo-wschodni','południowo-południowo-wschodni','południowy','południowo-południowo-zachodni','południowo-zachodni','zachodnio-południowo-zachodni','zachodni','zachodnio-północno-zachodni','północno-zachodni','północno-północno-zachodni'];
 const sectors=codes.map((code,i)=>({code,name:names[i],seconds:0,bins:[0,0,0,0,0],percent:null,binPercent:[0,0,0,0,0]}));
 let validSeconds=0,calmSeconds=0,missingDirectionSeconds=0;
 const records=payload.windRecords||[];
 for(let i=0;i<records.length;i++){
  const [ts,speed,direction]=records[i],next=records[i+1]?.[0]??Math.min(payload.serverTime,ts+60);
  const duration=Math.max(0,Math.min(end,next,ts+120)-Math.max(start,ts));
  if(!duration||speed==null||speed<0)continue;
  if(speed<1.8){calmSeconds+=duration;validSeconds+=duration;continue;}
  if(direction==null||direction<0||direction>360){missingDirectionSeconds+=duration;continue;}
  const sector=sectors[Math.round(direction/22.5)%16],bin=speed<5?0:speed<10?1:speed<20?2:speed<30?3:4;
  sector.seconds+=duration;sector.bins[bin]+=duration;validSeconds+=duration;
 }
 for(const sector of sectors){sector.percent=validSeconds?sector.seconds/validSeconds*100:null;sector.binPercent=sector.bins.map(n=>validSeconds?n/validSeconds*100:0);}
 const dominant=sectors.reduce((best,s)=>s.seconds>(best?.seconds||0)?s:best,null);
 return {start,end,sectors,validSeconds,missingDirectionSeconds,dominant,calmPercent:validSeconds?calmSeconds/validSeconds*100:null,coverage:Math.min(100,validSeconds/Math.max(1,end-start)*100)};
}
function loadStationRose(){
 if(!payload||!$('wind-rose'))return;
 // Backend supplies the exact Warsaw midnight, including DST.
 const dayStart=payload.dayStart;
 const periodStart=view?view[0]:payload.start,periodEnd=view?view[1]:payload.end;
 paintStationRoseData({today:csvWindRose(dayStart,payload.serverTime),period:csvWindRose(periodStart,periodEnd),method:''});
}
mountStationInstruments();
document.addEventListener('visibilitychange',()=>{if(document.hidden){if(sockFrame)cancelAnimationFrame(sockFrame);sockFrame=0;}else startStationSock();});
if(window.matchMedia)window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',startStationSock);
