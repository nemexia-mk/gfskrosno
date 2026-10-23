const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('station-dashboard/assets/dashboard.js','utf8');
const ext=source.slice(source.indexOf('const stationWindColors='),source.indexOf('function mountStationInstruments'));
vm.runInThisContext(ext);
assert.equal(stationRainScale(null).known,false);assert.equal(stationRainScale(0).known,true);
for(const x of [0,1,5,5.1,10,30,100,501,2001]){const s=stationRainScale(x);assert(s.max>=x);assert.equal(s.fraction,x/s.max);}
function tip(speed,phase=0){const nums=stationSockShape(speed,phase).at(-1).match(/-?\d+(?:\.\d+)?/g).map(Number);return {x:nums[2],y:(nums[3]+nums[5])/2};}
assert(tip(50).x>tip(10).x);assert(tip(10).x>tip(0).x);assert(tip(0).y>tip(50).y);
assert(Math.abs(tip(50,1).y-tip(50,2).y)>Math.abs(tip(5,1).y-tip(5,2).y));
assert.deepEqual(stationSockShape(null),stationSockShape(0));
assert(source.includes('renderTable();queueStationRose()'));assert(source.includes('updateStationInstruments(d);'));
console.log('PASS: wind shape, flutter, missing readings, rain scales, integration hooks');
fs.writeFileSync('/tmp/station-sock-preview.svg','<svg xmlns="http://www.w3.org/2000/svg" width="800" height="230" style="background:#102530">'+[0,10,50].map((speed,i)=>'<g transform="translate('+(45+i*260)+' 45)"><path d="M0 0V160" stroke="#bacbd0" stroke-width="4"/>'+stationSockShape(speed,1).map((d,j)=>'<path d="'+d+'" fill="'+(j%2?'#f0f7f5':'#ff9d57')+'"/>').join('')+'<text y="185" fill="white">'+speed+' km/h</text></g>').join('')+'</svg>');
