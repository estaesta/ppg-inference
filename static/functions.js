/* The column headings and description are:
_RAW files

PPGsum - This is the sum of the 3 Verity Sense light sensors. See Javascript line: PPGsum[i]=NewPPG[0][i]+NewPPG[1][i]+NewPPG[2][i]

PPGsum_f - PPGsum filtered/smoothed with a FIR (see: smoothPPG(devicename,dataobject,[0.025,0.04,0.07,0.13,0.2,0.22,0.2,0.13,0.07,0.04,0.025]))

PPG_f_reduced - PPGsum but with the number of datapoints reduced by a Ramer–Douglas–Peucker (RDP) algorithm that linearly interpolates values to the resolution displayed. This speeds up the display as plotting PPG_f could slow data collection. The trace is also useful when attempting to plot long time series using Excel etc.

PPGrate - rate of change of PPGsum_f (see: ratePPG(devicename,data))

PPGacceleration - rate of change of PPGrate (see: ratePPG(devicename,data) for the duration of the acceleration calculation - currently 4 data points to reduce noise)

PPGrate_reduced - RDP reduced PPGrate used for display during data collection and again use for plotting in Excel.

PPGpeak - List of PPG peaks see: function detectPPGpeak(devicename,data) for the algorithm and assumptions (e.g. max heart rate which might introduce issues under some circumstances)

PPGstart and PPGend - 'start' and 'end' of PPG waveform based on thresholds, see: function getPPGStart(data,minrate) and derivation of minrate
*/

lineformat = [];
newline = "\r\n";
defineLineformats();
Stream_Types = ["ECG", "PPG", "ACC", , , , , , , "SDK"];
Control_Types = [
  "",
  "Settings request",
  "Start measurement",
  "Stop measurement",
];
Device_Errors = [
  "Success",
  "Not supported",
  "Stream not known",
  "Stream not supported",
  "Invalid length",
  "Invalid parameter",
  "Already doing it!",
  "Invalid Resolution",
  "Invalid sample rate",
  "Invalid range",
  "Invalid MTU",
  "Invalid channels",
  "Invalid state",
  "Sorry, charging",
];
download_ignore_time = [
  "PPGsum_f",
  "acc_y",
  "acc_z",
  "acc_v",
  "timingend",
  "timing_smooth",
  "timingstart",
];
derived_data = ["PPGpeak", "PPGstart", "PPGend", "QRS"];
acc_range = 0x04; //hex(2) for range of 2G - 4 and 8G available
H10_rate = 0xc8; //hex(200) for sampling freqency of 200Hz  [H10]
Sense_rate = 416; //0xA0;//acc2_rate=208;//26Hz, 52Hz, 104Hz, 208Hz, 416Hz  [Verity Sense]
acc_resolution = 0x10; //hex(16) 16bit resolution
var samplerate = 44; //PPG maximum in SDK 28Hz, 44Hz, 135Hz, 176Hz
dataobject = {};
layoutCombined = [];
obj = [];
plotlyOut = 0;
var updatecombined = [];
//I need to sort out graphing times so that each graph just displays the last x seconds of data
const streams = {
  ACC_H10: {
    name: "ACC_H10",
    id: 2,
    type: "Accelerometer",
    code_start: [
      0x02,
      0x01,
      acc_range,
      0x00,
      0x00,
      0x01,
      H10_rate,
      0x00,
      0x01,
      0x01,
      acc_resolution,
      0x00,
    ],
  },
  SDK: { name: "SDK", id: 9, type: "SDK mode", code_start: [] },
  PPG: {
    name: "PPG",
    id: 1,
    type: "photoplethysmogram",
    code_start: [
      0x00,
      0x01,
      samplerate,
      0x00,
      0x01,
      0x01,
      0x16,
      0x00,
      0x04,
      0x01,
      0x04,
    ],
  },
  ACC_Sense: {
    name: "ACC_Sense",
    id: 2,
    type: "Accelerometer",
    code_start: [
      0x02,
      0x01,
      acc_range,
      0x00,
      0x00,
      0x01,
      0xa0,
      0x01,
      0x01,
      0x01,
      acc_resolution,
      0x00,
      4,
      1,
      3,
    ],
  },
  ECG: {
    name: "ECG",
    id: 0,
    type: "Electrocardiogram",
    code_start: [0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0e, 0x00],
  },
};

PMD_SERVICE = "fb005c80-02e7-f387-1cad-8acd2d8df0c8";
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8";
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8";
var traces = [];
var PPGvalues = [];
yscale_ECG_default = 2000;
yscale_PPG_default = 2000; // 30000
yscale = [];
datastore = [[], [], []];
//var layoutCombined={width:300,height:300,margin:{l:0,r:10,b:0,t:10,pad:10}, xaxis: {showticklabels:false}, yaxis: {showticklabels:false},yaxis2: {showticklabels:false,overlaying: 'y',side: 'top'}};
var layoutCombinedg = {
  autosize: false,
  width: 400,
  height: 150,
  margin: { l: 0, r: 10, b: 0, t: 0, pad: 0 },
  xaxis: { showticklabels: false },
  yaxis: { showticklabels: false },
  legend: { orientation: "h", font: { family: "sans-serif", size: 8 } },
};

layoutCombinedg.yaxis.range = [-yscale_PPG_default, yscale_PPG_default];
const devices = {
  Sense: { type: "Sense", output: ["SDK", "PPG", "ACC_Sense"] },
  H10: { name: "H10", output: ["ECG", "ACC_H10"] },
};
function addTrace(device, id) {
  traces[device][id] = { ...lineformat[id] };
  traces[device][id]["x"] = [];
  traces[device][id]["y"] = [];
}
function defineLineformats() {
  red = "255,0,0";
  cyan = "0,255,255";
  lineformat.ECG = defineLine(red, "ECG", "lines"); //lineformat.ECG.yaxis='y2';
  lineformat.ECG_reduced = defineLine(red, "ECG_reduced", "lines"); //lineformat.ECG_reduced.yaxis='y2';
  lineformat.acc_x = defineLine(red, "x", "lines");
  lineformat.acc_y = defineLine("0,255,0", "y", "lines");
  lineformat.acc_z = defineLine("0,255,0", "z", "lines");
  lineformat.PPG0 = defineLine(red, "0", "lines");
  lineformat.PPG1 = defineLine("0,255,0", "1", "lines");
  lineformat.PPG2 = defineLine("0,0,255", "2", "lines");
  lineformat.PPG3 = {
    type: "scattergl",
    mode: "lines",
    line: { color: "255,0,255", width: 1 },
    name: "ambient",
    x: [],
    y: [],
    yaxis: "y2",
  };
  lineformat.PPGsum = defineLine(cyan, "PPG", "lines");
  lineformat.acc_v = defineLine("0,0,255", "v_H10", "lines");
  lineformat.acc_v_reduced = defineLine("0,0,255", "acc_v_reduced", "lines");
  lineformat.QRS = defineLine("75,0,75", "QRS", "markers"); //lineformat.QRS.yaxis='y2';
  lineformat.PPGpeak = defineLine("75,0,75", "PPGpeak", "markers");
  lineformat.PPGstart = defineLine("75,75,75", "PPGstart", "markers");
  lineformat.PPGend = defineLine("0,75, 75", "PPGend", "markers");
  lineformat.PPGsum_f = defineLine("0,128,128", "PPGsum_f", "lines");
  lineformat.PPG_f_reduced = defineLine("0,255,0", "PPG_f_reduced", "lines");

  lineformat.PPGrate = defineLine("128,0,0", "PPGrate", "lines");
  lineformat.PPGacceleration = defineLine("0,0,0", "PPGrate", "lines");
  lineformat.PPGrate_reduced = defineLine(
    "0,128,0",
    "PPGrate_reduced",
    "lines",
  );
  lineformat.timing = defineLine("128,128,128", "Interval", "markers");
  lineformat.timing.marker = { opacity: 0.5, size: 5 };
  lineformat.timingstart = defineLine("75,128,75", "Interval", "markers");
  lineformat.timingend = defineLine("255,128,75", "Interval", "markers");
  lineformat.timing_smooth = defineLine("255,0,0", "Interval", "lines");
}

let acc_outputfile = "";
timeconstant = 0.2; //timeconstant for highpass in s
duration = 5000; //ms for graph
starttime = [];
last_filter = [];
last_incom = {};
intime = [];
var signedconvert = [];
signedconvert[0] = 0;
signedconvert[255] = 2 ** 16;

function alterGain(name, fraction) {
  console.log(layoutCombined[name].yaxis.range[0]);
  layoutCombined[name].yaxis.range = [
    layoutCombined[name].yaxis.range[0] * fraction,
    layoutCombined[name].yaxis.range[1] * fraction,
  ];
}
ecg_timestep = 1000 / 130;
H10_rate = 0xc8;
acc_timestep = 1000 / H10_rate;

async function addDevice() {
  num = Object.keys(obj) + 1;
  newdevice = await navigator.bluetooth.requestDevice({
    filters: [{ namePrefix: "Polar" }],
    acceptAllDevices: false,
    manufacturerData: [{ companyIdentifier: 0x00d1 }],
    optionalServices: [PMD_SERVICE],
  });
  name = newdevice.name;
  devicetype = name.split(" ")[1];
  obj[num] = structuredClone(devices[devicetype]); //create alias to devices[name]
  obj[num]["device"] = newdevice;
  var mydiv = document.getElementById(name);
  if (mydiv === null) {
    await obj[num].device.addEventListener(
      "gattserverdisconnected",
      onDisconnected,
    );
    obj[num].server = await obj[num].device.gatt.connect();
    obj[num].service = await obj[num].server.getPrimaryService(PMD_SERVICE);
    obj[num].character = await obj[num].service.getCharacteristic(PMD_CONTROL);
    // controlfunction = await obj[num].character.addEventListener(
    //   "characteristicvaluechanged",
    //   printcontrolvalue,
    // );
    await obj[num].character.startNotifications();
    obj[num].data = await obj[num].service.getCharacteristic(PMD_DATA);
    await obj[num].data.startNotifications();
    outputfunction = await obj[num].data.addEventListener(
      "characteristicvaluechanged",
      printHeartRate,
    );
    // serviceButtons(name, devicetype);
    // init echart
    Object.defineProperty(obj, name, Object.getOwnPropertyDescriptor(obj, num));
    delete obj[num];
    layoutCombined[name] = JSON.parse(JSON.stringify(layoutCombinedg));
    if (devicetype == "Sense") {
      layoutCombined[name].yaxis.range = [
        -yscale_PPG_default,
        yscale_PPG_default / 3,
      ];
    } else {
      layoutCombined[name].yaxis.range = [
        -yscale_ECG_default,
        yscale_ECG_default,
      ];
    }
  }
}
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
function ButtonColour(buttonID, colour, state) {
  id = document.getElementById(buttonID);
  id.style.backgroundColor = colour;
  id.disabled = !state;
}
async function ActivateStream(name, stream) {
  // let tickbox = document.getElementById(name + stream.name);
  if (obj[name].character !== undefined) {
    console.log("Activating stream ", stream.name);
    var Init = [0x02, stream.id];
    BytesToSend = Init.concat(stream.code_start);
    await obj[name].character.writeValue(new Uint8Array(BytesToSend));
    console.log("Stream ", stream.name, " activated");
    // tickbox.checked = true;
  } else {
    // tickbox.checked = false;
  }
}
async function AllStreamsOn(name) {
  for (s in obj[name].output) {
    await ActivateStream(name, streams[obj[name].output[s]]);
    await sleep(100);
  }
}
async function AllStreamsOff(name) {
  list = [...obj[name].output].reverse();
  for (s in list) {
    await streamOff(name, streams[list[s]]);
  }
}
async function connectStream(name, device, b) {
  state = document.getElementById(name + b).checked;
  stream = streams[b];
  if (state == true) {
    await ActivateStream(name, stream);
  } else {
    streamOff(name, stream);
  }
}
var refreshInterval;
async function start() {
  // check if the device is connected
  // if obj empty
  if (obj.length == 0) {
    alert("Please connect the device first");
    return;
  }
  time = document.getElementById("time").value;
  // check if the time is -1 (unlimited)
  if (time == -1) {
    //activate sdk and ppg
    await ActivateStream(name, streams["SDK"]);
    await sleep(100);
    await ActivateStream(name, streams["PPG"]);
    // change button from start to stop and chang the colour
    button = document.getElementById("start-button");
    button.innerHTML = "Stop";
    button.onclick = stop;
    button.className = "btn btn-danger";

    // start chart
    chart = echarts.init(document.getElementById("ecgplotchart"));
    refreshInterval = setInterval(() => {
      chart.setOption({
        series: [
          {
              // data: dataobject[name].PPGsum_f.y,
            data: dataBuffer,
          },
        ],
      });
    }, 1000);
  } else {
    //activate sdk and ppg
    await ActivateStream(name, streams["SDK"]);
    await sleep(100);
    await ActivateStream(name, streams["PPG"]);
     // initialize the timer
    let timer = time * 60;

    // update the timer every second
    // time left
    timerInterval = setInterval(() => {
      timer--;
      document.getElementById("timer").innerText = `Time left: ${timer} seconds`;
    }, 1000);   

    // change button from start to stop and chang the colour
    button = document.getElementById("start-button");
    button.innerHTML = "Stop";
    button.onclick = stop;
    button.className = "btn btn-danger";

    // start chart
    chart = echarts.init(document.getElementById("ecgplotchart"));
    refreshInterval = setInterval(() => {
      chart.setOption({
        series: [
          {
            // data: dataobject[name].PPGsum_f.y,
            data: dataBuffer,
          },
        ],
      });
    }, 1000);
    // add event listener to stop the timer
    document.addEventListener("stopped", () => {
      clearInterval(timerInterval);
      document.getElementById("timer").innerText = "Time left: 0 seconds";
    });
    // stop the chart after the time
    setTimeout(() => {
      stop();
    }, time * 1000 * 60);
  }
}
async function stop() {
  //deactivate sdk and ppg
  await streamOff(name, streams["PPG"]);
  await sleep(100);
  await streamOff(name, streams["SDK"]);
  // change button from stop to start and chang the colour
  button = document.getElementById("start-button");
  button.innerHTML = "Start";
  button.onclick = start;
  button.className = "btn btn-success";
  // add an event (stopped)
  var event = new Event("stopped");
  document.dispatchEvent(event);
  clearInterval(refreshInterval);
}
async function StartStreams() {
  time = [];
  for (const [name, device] of Object.entries(devices)) {
    for (const stream of Object.entries(device.output)) {
      await ActivateStream(device, streams[stream[1]]);
    }
  }
  ButtonColour("Start streams", "Green", false);
  ButtonColour("Stop streams", "", true);
}
async function streamOff(name, stream) {
  off = new Uint8Array([0x03, stream.id]);
  a = await obj[name].character.writeValue(off);
  // document.getElementById(name + stream.name).checked = false;
}
async function disconnect() {
  console.log("disconnect()");
  for (const [name, device] of Object.entries(devices)) {
    if ("server" in device) {
      for (const stream of Object.entries(device.output).reverse()) {
        await streamOff(device, streams[stream[1]]);
      }
    }
  }
}
// flag
function pushDataTrace(device, name, x, y) {
  if (!(name in traces[device])) {
    addTrace(device, name);
  }
  dataobject[device][name].x.push(x);
  traces[device][name].x.push(x);
  dataobject[device][name].y.push(y);
  traces[device][name].y.push(y);
}
function pushData(device, ids, timearray, y_values) {
  if (!(device in dataobject)) {
    dataobject[device] = {};
  }
  if (!(device in traces)) {
    traces[device] = {};
  }
  ids.forEach(function (value) {
    if (!(value in dataobject[device])) {
      dataobject[device][value] = { x: [], y: [] };
    }
    dataobject[device][value].x.push(...timearray);
    dataobject[device][value].y.push(...y_values[ids.indexOf(value)]);
    if (!(value in traces[device])) {
      addTrace(device, value);
    }
    traces[device][value].x.push(...timearray);
    traces[device][value].y.push(...y_values[ids.indexOf(value)]);
  });
}
function DataArrayLength(devicename, name) {
  if (dataobject[devicename][name] == undefined) {
    dataobject[devicename][name] = { x: [], y: [] };
    return 0;
  } else {
    return dataobject[devicename][name].x.length;
  }
}
function completeDeltaFrame(data, num_chan, bytes) {
  headerpointer = 10 + num_chan * bytes;
  framepointer = headerpointer + 2;
  data_array = getInitialSensorValues(data.slice(10, headerpointer), bytes);
  while (framepointer < data.byteLength) {
    DeltaFrameDetails = DeltaFrameDescription(
      data.slice(headerpointer, framepointer),
      num_chan,
    );
    nextheaderpointer = framepointer + DeltaFrameDetails.bytes;
    Frame = reSlice(
      ChunkArray(data.slice(framepointer, nextheaderpointer)),
      DeltaFrameDetails.bits,
      DeltaFrameDetails.channels,
    );
    data_array = addDeltaframe(Frame, data_array);
    headerpointer = nextheaderpointer;
    framepointer = headerpointer + 2;
  }
  return data_array;
}
function defineLine(color, name, linetype) {
  return {
    type: "scattergl",
    mode: linetype,
    showlegend: true,
    line: { color: "rgb(" + color + ")", width: 1 },
    name: name,
    x: [],
    y: [],
  };
}

// flag
function serviceButtons(name, devicetype) {
  console.log("Adding service buttons for ", name);
  output = "<br><table border=1><tr>";
  for (const [device, variables] of Object.entries(devices)) {
    if (device == devicetype) {
      output += "<td>" + name + ": ";
      for (const [stream, v] of Object.entries(variables.output)) {
        output +=
          "&nbsp&nbsp" +
          "<input type='checkbox' id='" +
          name +
          v +
          "' name='" +
          v +
          "' onClick='connectStream(\"" +
          name +
          '","' +
          device +
          '","' +
          v +
          "\")'>" +
          v +
          "&nbsp&nbsp";
      }
    }
  }
  startbutton =
    " <button id='download' onClick='AllStreamsOn(\"" +
    name +
    "\")'>Start all streams</button> ";
  stopbutton =
    " <button id='download' onClick='AllStreamsOff(\"" +
    name +
    "\")'>Stop all streams</button> ";
  text =
    "<td><span id='" +
    name +
    "'>" +
    output +
    startbutton +
    stopbutton +
    "</span></td>";
  document.getElementById("streams").insertAdjacentHTML("afterend", text);
  if (devicetype == "Sense") {
    temptext = "PPGpeak size (kArb): ";
  } else {
    temptext = "R wave peak (mV): ";
  }
  document
    .getElementById("graphsdiv")
    .insertAdjacentHTML(
      "afterend",
      "<table><tr><td><span id='graph" +
        name +
        "'></span></td><td><table><tr><td><input type='button' value='+' onClick='alterGain(\"" +
        name +
        "\",0.8)'></td></tr><tr><td><input type='button' value='-' onClick='alterGain(\"" +
        name +
        "\",1.2)'></td></tr></table></td><td><table><tr></td>Heart rate=<span id='bpm" +
        name +
        "'></span></td></tr><tr><td>" +
        temptext +
        "<span id='size" +
        name +
        "'></span></td></tr></table></td></tr></table>",
    );
}
function fillTimeArray(Type, devicename, t, dTime, num, step) {
  start_packet_time = Number(new BigInt64Array(t)[0]) / 1000000 - num * step;
  computer_time = dTime - num * step;
  if (starttime[devicename] === undefined) {
    var x = new Date("1/1/2000 00:00:00");
    var y = new Date("1/1/1970 00:00:00");
    let seconds = Math.abs(x.getTime() - y.getTime()) + start_packet_time;
    startdate = new Date(seconds);
    console.log("Device started at ", startdate);
    if (firstTime == 0) {
      firstTime = computer_time;
    }
    deviceEPOCH = computer_time - start_packet_time;
    starttime[devicename] = start_packet_time; //deviceEPOCH-firstTime;
    console.log(
      devicename,
      " approx EPOCH for stream ",
      Stream_Types[Type],
      " Type ",
      Type,
      " ",
      new Date(deviceEPOCH).toISOString(),
    );
  }
  stream_time = start_packet_time; //+starttime[devicename];//try to switch to real time recording of time.
  a = createArray(num);
  for (i = 0; i < num; i++) {
    a[i] = Math.round((stream_time + step * i) * 1000) / 1000;
  }
  return a;
}

function createArray(length) {
  var arr = new Array(length || 0),
    i = length;
  if (arguments.length > 1) {
    var args = Array.prototype.slice.call(arguments, 1);
    while (i--) arr[length - 1 - i] = createArray.apply(this, args);
  }
  return arr;
}
function WordstoSignedInteger(words, BitsPerWord) {
  val = 0;
  word_val = 2 ** BitsPerWord;
  for (i = 0; i < words.length; i += 1) {
    val += words[i] * word_val ** i;
  }
  bits = words.length * BitsPerWord;
  if (val > 2 ** (bits - 1)) {
    val = val - 2 ** bits;
  }
  return val;
}
function getInitialSensorValues(a, bytes) {
  a = new Uint8Array(a);
  sensors = createArray(a.length / bytes, 1);
  offset = 0;
  while (offset < a.length) {
    sensors[offset / bytes][0] = WordstoSignedInteger(
      a.slice(offset, offset + bytes),
      8,
    );
    offset += bytes;
  }
  return sensors;
}
function addDeltaframe(frame, data_array) {
  chans = frame[0].length;
  for (offset = 0; offset < frame.length; offset += 1) {
    for (ch = 0; ch < chans; ch += 1) {
      data_array[ch].push(+data_array[ch].slice(-1) + frame[offset][ch]);
    }
  }
  return data_array;
}
function DeltaFrameDescription(b, channels) {
  a = new Uint8Array(b);
  bits = a[0];
  number = a[1];
  return {
    bits: bits,
    number: number,
    bytes: Math.ceil(((number * bits) / 8) * channels),
    channels: channels,
  };
}
function ChunkByte(sbyte) {
  n = [];
  ts = [1, 4, 16, 64];
  tg = [3, 12, 48, 192];
  for (a = 0; a < 4; a += 1) {
    n[a] = (sbyte & tg[a]) / ts[a];
  }
  return n;
}
function ChunkArray(arr) {
  arr = new Uint8Array(arr);
  offset = 0;
  NewArray = [];
  while (offset < arr.length) {
    NewArray.push(...ChunkByte(arr[offset]));
    offset += 1;
  }
  return NewArray;
}
function HPfilter(array, last_in, last_out) {
  f_arr = createArray(array.length);
  fraction = 1 + 1 / (samplerate * timeconstant);
  array.forEach(function (value, n) {
    last_out = f_arr[n] = Math.round((value + last_out - last_in) / fraction);
    last_in = value;
  });
  return f_arr;
}
function reSlice(arr, bits, channels) {
  offset = 0;
  block = (bits / 2) * channels;
  len = Math.floor(arr.length / block);
  f_array = createArray(len, channels);
  while (offset < len * block) {
    for (a = 0; a < channels; a += 1) {
      mini_array = arr.slice(
        offset + (a * bits) / 2,
        offset + ((a + 1) * bits) / 2,
      );
      f_array[offset / block][a] = WordstoSignedInteger(mini_array, 2);
    }
    offset += block;
  }
  return f_array;
}
function onDisconnected(event) {
  const device = event.target;
  var name = device.name;
  console.log("onDisconnect", event);
  div = document.getElementById(name);
  if (div != null) {
    div.parentNode.removeChild(div);
    div = document.getElementById("graph" + name);
    div.parentNode.removeChild(div);
  }
  document.getElementById(name).innerHTML = "";
  console.log(`Device ${device.name} is disconnected.`);
}
function GrabTiming() {
  document.getElementById("confirmed").value =
    Math.round(
      dataobject["timing_smooth"].y[dataobject["timing_smooth"].y.length - 1] *
        10,
    ) / 10;
  document.getElementById("confirmed_BPM").value =
    document.getElementById("bpm").innerHTML;
}
function die(str) {
  throw new Error(str || "Script ended by death");
}
function report(obj) {
  console.log("Obj - ", obj);
}
function printcontrolvalue(event) {
  devicename = event.currentTarget.service.device.name;
  response = new Uint8Array(event.currentTarget.value.buffer);
  console.log(
    devicename,
    "(timeStamp=",
    event.timeStamp,
    ") ",
    Control_Types[response[1]],
    Stream_Types[response[2]],
    Device_Errors[response[3]],
  );
}
firstTime = 0;
graph_time = [];
dataoutput = "";
function resetTraces(frame, ids) {
  ids.forEach(function (value) {
    if (traces[value].x.length > 0) {
      traces[value].x = [];
      traces[value].y = [];
    }
  });
}
function download() {
  lists = [];
  for (a in dataobject) {
    for (let key of Object.keys(dataobject[a])) {
      lists.push(key);
    }
  }
  var csvdata = new Blob([DataObjectToCSV(lists)]);
  var a = window.document.createElement("a");
  a.href = window.URL.createObjectURL(csvdata, { type: "text/plain" });
  a.download = "Polar data " + new Date(firstTime).toISOString() + "_RAW.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  var csvdata = new Blob([DataObjectToCSV(derived_data)]);
  a.href = window.URL.createObjectURL(csvdata, { type: "text/plain" });
  a.download = "Polar data " + new Date(firstTime).toISOString() + "_peaks.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
function emptybuffer() {
  dataobject = {};
  traces = {};
  dataBuffer = [];
  initChart();
}
function copyTable() {
  ttext = document.getElementById("ttable").innerHTML;
  navigator.clipboard.writeText(ttext);
}
function FindLastPeak(data, threshold) {
  for (i = data.x.length; i > 0; i--) {
    if (data.y[i] > threshold) {
      peak = Math.max(...data.y.slice(i - 20, i));
      peakindex = data.y.slice(i - 20, i).indexOf(peak);
      break;
    }
  }
  return data.x[i - 20 + peakindex];
}
function ReduceOld(devicename, data, sourcename, targetname, resolution) {
  if (data[devicename][targetname] == undefined) {
    data[devicename][targetname] = { x: [], y: [] };
    addTrace(devicename, targetname);
    start_index = 0;
  } else {
    start_index = data[devicename][sourcename].x.indexOf(
      data[devicename][targetname].x[data[devicename][targetname].x.length - 1],
    );
    array = {
      x: data[devicename][sourcename].x.slice(
        start_index,
        data[devicename][sourcename].x.length,
      ),
      y: data[devicename][sourcename].y.slice(
        start_index,
        data[devicename][sourcename].y.length,
      ),
    };
    t = RDP(array, resolution);
    data[devicename][targetname].x.push(...t.x);
    data[devicename][targetname].y.push(...t.y);
    traces[devicename][targetname].x.push(...t.x);
    traces[devicename][targetname].y.push(...t.y);
  }
}
function Reduce(devicename, data, sourcename, targetname, resolution) {
  if (data[devicename][targetname] == undefined) {
    data[devicename][targetname] = { x: [], y: [] };
    source = data[devicename][sourcename];
    target = data[devicename][targetname];
    start_index = 0;
  } else {
    start_index = source.x.indexOf(target.x[target.x.length - 1]);
    array = {
      x: source.x.slice(start_index, source.x.length),
      y: source.y.slice(start_index, source.y.length),
    };
    t = RDP(array, resolution);
    data[devicename][targetname].x.push(...t.x);
    data[devicename][targetname].y.push(...t.y);
    if (!(targetname in traces[devicename])) {
      addTrace(devicename, targetname);
    }
    traces[devicename][targetname].x.push(...t.x);
    traces[devicename][targetname].y.push(...t.y);
  }
}
function RDP(data, resolution) {
  var dmax = 0;
  var maxv = 0;
  var end_index = data.x.length - 1;
  var m = (data.y[0] - data.y[end_index]) / (data.x[0] - data.x[end_index]);
  var c = data.y[0] - m * data.x[0];
  for (var j = 1; j < end_index; j++) {
    var error = Math.abs(data.y[j] - m * data.x[j] - c);
    if (error > dmax) {
      dmax = error;
      maxv = j;
    }
  }
  var r1 = { x: [], y: [] };
  if (dmax > resolution) {
    end_index++;
    cut = maxv + 1;
    r1 = RDP({ x: data.x.slice(0, cut), y: data.y.slice(0, cut) }, resolution);
    var r2 = RDP(
      { x: data.x.slice(maxv, end_index), y: data.y.slice(maxv, end_index) },
      resolution,
    );
    r1.x.push(...r2.x);
    r1.y.push(...r2.y);
  } else {
    r1.x[0] = data.x[0];
    r1.y[0] = data.y[0];
  }
  return r1;
}

function detectQRS(data, devicename) {
  min_samples_betweenQRS = 10;

  numQRS = DataArrayLength(devicename, "QRS");
  if (numQRS > 0) {
    i = data.ECG.x.indexOf(data.QRS.x[numQRS - 1]) + min_samples_betweenQRS;
  } else {
    i = 0;
  }
  while (i < data.ECG.x.length) {
    next = i + 1;
    rate =
      (data.ECG.y[next] - data.ECG.y[i]) / (data.ECG.x[next] - data.ECG.x[i]);
    //if ECG is faster than -60uV/ms
    if (rate < -60) {
      ECGmax = Math.max(...data.ECG.y.slice(i - 10, next));
      peakindex = data.ECG.y.slice(i - 10, next).indexOf(ECGmax);
      //flag
      pushDataTrace(devicename, "QRS", data.ECG.x[i - 10 + peakindex], ECGmax);
      i += min_samples_betweenQRS;
    } else {
      i += 1;
    }
  }
  if (dataobject[devicename]["QRS"].x.length > 5) {
    document.getElementById("bpm" + devicename).innerHTML = Math.round(
      (5 * 60000) /
        (dataobject[devicename]["QRS"].x[
          dataobject[devicename]["QRS"].x.length - 1
        ] -
          dataobject[devicename]["QRS"].x[
            dataobject[devicename]["QRS"].x.length - 6
          ]),
    );
    document.getElementById("size" + devicename).innerHTML =
      Math.round(
        dataobject[devicename]["QRS"].y[
          dataobject[devicename]["QRS"].y.length - 1
        ] / 10,
      ) / 100;
  }
}
function detectPPGpeak(devicename, data) {
  maximum_heart_rate = 170;
  min_samples_betweenPPGpeaks = Math.round(
    (60 / maximum_heart_rate) * samplerate,
  );
  triggerrate =
    (layoutCombined[devicename].yaxis.range[1] -
      layoutCombined[devicename].yaxis.range[0]) /
    20;
  numPPGpeak = DataArrayLength(devicename, "PPGpeak");
  DataArrayLength(devicename, "PPGstart");
  DataArrayLength(devicename, "PPGend");
  if (numPPGpeak > 0) {
    i =
      data[devicename].PPGsum_f.x.indexOf(
        data[devicename].PPGpeak.x[numPPGpeak - 1],
      ) + min_samples_betweenPPGpeaks;
  } else {
    i = 0;
  }
  while (i < data[devicename].PPGsum_f.x.length - 40) {
    if (data[devicename].PPGrate.y[i] < -triggerrate) {
      PPGratemin = Math.min(...data[devicename].PPGrate.y.slice(i, i + 20));
      peakindex = data[devicename].PPGrate.y
        .slice(i, i + 20)
        .indexOf(PPGratemin);
      pushDataTrace(
        devicename,
        "PPGpeak",
        data[devicename].PPGrate.x[peakindex + i],
        PPGratemin,
      );
      startindex = getPPGStart(
        data[devicename].PPGrate.y.slice(i - 20, i),
        PPGratemin,
      );
      endindex = getPPGend(
        data[devicename].PPGrate.y.slice(i + peakindex, i + 30 + peakindex),
        PPGratemin,
      );
      pushDataTrace(
        devicename,
        "PPGstart",
        data[devicename].PPGrate.x[startindex - 20 + i],
        data[devicename].PPGrate.y[startindex - 20 + i],
      );
      pushDataTrace(
        devicename,
        "PPGend",
        data[devicename].PPGrate.x[endindex + i + peakindex],
        data[devicename].PPGrate.y[endindex + i + peakindex],
      );
      //the issue is that the other devices might not have reported that data yet, so need to set a trigger time of the first device and then wait for the others to pass it. i.e. reference beat.
      //the logic is store a reference beat - wait for the others to pass it by 1s and then if a match isn't found move to the next reference beat.
      //or save a reference beat and then look in traces with later times. Need a reference beat for each not first device.
      console.log(
        "find matching for previous peak  time " +
          (data[devicename].PPGrate.x[peakindex + i] -
            data[devicename].PPGrate.x[peakindex]) +
          " ms ago",
      );
      if (Object.keys(dataobject)[0] == devicename) {
        console.log(dataobject[0], devicename);
      } else {
        console.log("not");
      }
      i += min_samples_betweenPPGpeaks;
    } else {
      i += 1;
    }
  }
  document.getElementById("bpm" + devicename).innerHTML = Math.round(
    (5 * 60000) /
      (dataobject[devicename]["PPGpeak"].x[
        dataobject[devicename]["PPGpeak"].x.length - 1
      ] -
        dataobject[devicename]["PPGpeak"].x[
          dataobject[devicename]["PPGpeak"].x.length - 6
        ]),
  );
  document.getElementById("size" + devicename).innerHTML =
    Math.round(
      dataobject[devicename]["PPGpeak"].y[
        dataobject[devicename]["PPGpeak"].y.length - 1
      ] / 100,
    ) / 10;
}

function getPPGStart(data, minrate) {
  var i = data.length - 1;
  var threshold = minrate / 5;
  while (data[i] < threshold) {
    i -= 1;
  }
  return i;
}

function getPPGend(data, minrate) {
  var i = 0;
  var threshold = minrate / 5;
  while (data[i] < threshold) {
    i += 1;
  }
  return i;
}

function getTimings(data) {
  var update = false;
  if (data.PPGpeak !== undefined && data.QRS !== undefined) {
    start_index = DataArrayLength("timing");
    DataArrayLength("timingstart");
    DataArrayLength("timing_smooth");
    DataArrayLength("timingend");
    for (i = start_index; i < data.QRS.x.length - 1; i++) {
      QRSpeaktime = data.QRS.x[i];
      lastPPG = data.PPGpeak.x.length;
      while (data.PPGpeak.x[--lastPPG] > QRSpeaktime);
      var PPGtime = data.PPGpeak.x[lastPPG + 1];
      if (PPGtime !== undefined) {
        update = true;
        delay = PPGtime - QRSpeaktime;
        delaystart = data.PPGstart.x[lastPPG + 1] - QRSpeaktime;
        delayend = data.PPGend.x[lastPPG + 1] - QRSpeaktime;
        pushDataTrace("timing", QRSpeaktime, delay);
        pushDataTrace("timingstart", QRSpeaktime, delaystart);
        pushDataTrace("timingend", QRSpeaktime, delayend);
        if (i == 0) {
          smoothdelay = delay;
        } else {
          smoothdelay =
            data.timing_smooth.y[i - 1] +
            (delay - data.timing_smooth.y[i - 1]) * 0.05;
        }
        if (
          Math.abs((smoothdelay - delay) / delay) > 0.15 ||
          smoothdelay == undefined
        ) {
          smoothdelay = delay;
        }
        document.getElementById("timing_smooth").innerHTML =
          Math.round(smoothdelay * 10) / 10;
        pushDataTrace("timing_smooth", QRSpeaktime, smoothdelay);
      }
    }
  }
  return update;
}

function AddNotes() {
  startindex = DataArrayLength("Systolic");
  DataArrayLength("Diastolic");
  DataArrayLength("Notes");
  eventtime = dataobject["timing"].x[dataobject["timing"].x.length - 1];
  var systolic = document.getElementById("systolic").value;
  var diastolic = document.getElementById("diastolic").value;
  var notes = document.getElementById("notes").value;
  var bpm = document.getElementById("confirmed_BPM").value;
  var entrytime = new Date();
  var ISOtime = entrytime.toISOString();
  entrytime = entrytime / 1000 / 24 / 60 / 60 + 25569;
  var timingsmooth = document.getElementById("confirmed").value;
  dataobject["Systolic"].x[startindex] = eventtime;
  dataobject["Diastolic"].x[startindex] = eventtime;
  dataobject["Notes"].x[startindex] = eventtime;
  dataobject["Systolic"].y[startindex] = systolic;
  dataobject["Diastolic"].y[startindex] = diastolic;
  dataobject["Notes"].y[startindex] = notes;
  document.getElementById("confirmed").value =
    timingsmooth + "ms at " + eventtime + "ms";
  document.getElementById("list_timing").innerHTML =
    document.getElementById("list_timing").innerHTML +
    "<tr><td>" +
    ISOtime +
    "</td><td>" +
    entrytime +
    "</td><td>" +
    systolic +
    "</td><td>" +
    diastolic +
    "</td><td>" +
    timingsmooth +
    "</td><td>" +
    bpm +
    "</td><td>" +
    notes +
    "</td></tr>";
}

function ratePPG(devicename, data) {
  start_index = DataArrayLength(devicename, "PPGrate");
  DataArrayLength(devicename, "PPGacceleration");
  time = 1000 / samplerate; //this is a cludge because of the dodgy timing
  for (i = start_index; i < data[devicename].PPGsum_f.x.length; i++) {
    rate = Math.round(
      ((data[devicename].PPGsum_f.y[i] - data[devicename].PPGsum_f.y[i - 1]) /
        time) *
        100,
    );
    pushDataTrace(devicename, "PPGrate", data[devicename].PPGsum_f.x[i], rate);
    acceleration = Math.round(
      (((data[devicename].PPGrate.y[i] - data[devicename].PPGrate.y[i - 3]) /
        time) *
        100) /
        4,
    );
    pushDataTrace(
      devicename,
      "PPGacceleration",
      data[devicename].PPGsum_f.x[i - 1],
      acceleration,
    );
  }
}

function smoothPPG(devicename, data, filter) {
  // console.log("devicename",devicename,"smoothing",data);
  var sumfilter = 0;
  for (var i = filter.length; i--; ) {
    sumfilter += filter[i];
  }
  filter_offset = (filter.length - 1) / 2;
  start_index = DataArrayLength(devicename, "PPGsum_f");
  if (start_index == 0) {
    for (i = 0; i < filter_offset; i++) {
      data[devicename].PPGsum_f.x[i] = data[devicename].PPGsum.x[i];
      data[devicename].PPGsum_f.y[i] = data[devicename].PPGsum.y[i];
    }
    start_index = filter_offset;
  }

  for (
    i = start_index;
    i < data[devicename].PPGsum.x.length - filter_offset;
    i++
  ) {
    smooth = 0;
    first_unfiltered = i - filter_offset;
    filter.forEach(function (value, n) {
      smooth += value * data[devicename].PPGsum.y[first_unfiltered + n];
    });
    smooth = Math.round(smooth / sumfilter);
    pushDataTrace(devicename, "PPGsum_f", data[devicename].PPGsum.x[i], smooth);
    // dataBuffer.push({ x: data[devicename].PPGsum.x[i], y: smooth });
    dataBuffer.push(smooth);
  }

  // if minimum number of samples are available, send data to API
  if (dataBuffer.length >= windowSize) {
    // keep only the last windowSize samples
    dataBuffer = dataBuffer.slice(-windowSize);
    // downsample to 64Hz
    // dataToSend = downsample(dataBuffer);
    // send data to API
    sendDataToAPI(dataBuffer);
  }
  //
  // // log the length of the ppgsum_f array
  // console.log("PPGsum_f length", data[devicename].PPGsum_f.x.length);
  // // log the diff or length of data added
  // console.log("PPGsum_f diff", data[devicename].PPGsum_f.x.length - start_index);
  //console.log(data.PPGsum.x.slice(start_index,start_index+8),data.PPGsum.y.slice(start_index,start_index+8), data.PPGsum_f.y.slice(start_index,start_index+8))
}

// send as { ppg_signal: [1, 2, 3, 4, 5, 6, 7, 8] }
async function sendDataToAPI(data) {
  try {
    let response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ppg_signal: data,
        model: document.getElementById("modelSelect").value,
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    let responseData = await response.json(); // Assuming the response is in JSON format
    console.log("Data sent successfully:", responseData);
    displayResponse(responseData); // Function to update the UI
  } catch (error) {
    console.error("Error sending data to API:", error);
  }
}

// async function changeModel(model) {
// // fetch to /predict post
//   try {
//     let response = await fetch("/change_model", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({ model: model }),
//     });
//     if (!response.ok) {
//       throw new Error(`HTTP error! status: ${response.status}`);
//     }
//     let responseData = await response.json(); // Assuming the response is in JSON format
//     console.log("Model changed successfully:", responseData);
//     // displayResponse(responseData); // Function to update the UI
//   }
//   catch (error) {
//     console.error("Error changing model:", error);
//   }
// }
// window.addEventListener("DOMContentLoaded", (event) => {
//   modelSelect = document.getElementById("modelSelect");
//   modelSelect.addEventListener("change", function () {
//     changeModel(modelSelect.value);
//   });
// });


// nats publish
async function publishDataToNats(data) {
  //TODO: Implement the nats publish logic
}

// downsample to 64Hz
function downsample(data) {
  let downsampledData = [];
  for (let i = 0; i < data.length; i += 2) {
    downsampledData.push(data[i]);
  }
  return downsampledData;
}

function displayResponse(responseData) {
    // result = {"result": result, "hr": hr[-1:], "mean_hr": np.mean(hr)}
  // parse the response data
  var { result, hr, mean_hr } = responseData;
  const responseContainer = document.getElementById('responseContainer');
  // Clear the container
  responseContainer.innerHTML = '';

  // Update the container with the response data
  // the result
  responseContainer.innerText = `Result: ${result}`;

  info = document.getElementById('info');
  // Update the container with the response data
  info.innerText = `HR: ${hr}, Mean HR: ${mean_hr}`;
}

function getCSVmarkers(list) {
  mark = document.getElementById("list_timing");
  markers = "";
  for (var i = 0, row; (row = mark.rows[i]); i++) {
    for (var j = 0, col; (col = row.cells[j]); j++) {
      markers += col.innerHTML + ",";
    }
    markers += "\r\n";
  }
  markers = "Markers" + newline + markers + newline;
  return markers;
}
function getCSVheadings(for_streams) {
  output = "";
  devicetext = "";
  headings = "";
  for (var datadevices in dataobject) {
    devicetext += ",";
    for (var stream in dataobject[datadevices]) {
      if (for_streams.includes(stream)) {
        headings += "Time," + stream + ",";
        devicetext += ",,";
      }
    }
    headings += ",";
    output +=
      datadevices +
      ": start time=" +
      getFirstTime(datadevices) +
      ",Excel datetime," +
      (getFirstTime(datadevices) / 1000 / 24 / 60 / 60 + 36526) +
      devicetext.substring(0, devicetext.length - 2);
  }
  output += newline + headings + newline;
  console.log(dataobject);
  return output;
}
function getFirstTime(datadevices) {
  firstkey = Object.values(dataobject[datadevices]);
  return firstkey[0].x[0];
}
function getCSVdata(for_streams) {
  output = "";
  bit = "";
  loop = 0;
  do {
    test = 0;
    for (var datadevices in dataobject) {
      for (var stream in dataobject[datadevices]) {
        if (for_streams.includes(stream)) {
          if (loop < dataobject[datadevices][stream].x.length) {
            bit +=
              Math.round(
                dataobject[datadevices][stream].x[loop] -
                  getFirstTime(datadevices),
              ) /
                1000 +
              "," +
              dataobject[datadevices][stream].y[loop] +
              ",";
            test = 1;
          } else {
            bit += ",,";
          }
        }
      }
      bit += ",";
    }
    bit += newline;
    loop += 1;
  } while (test == 1);
  console.log(dataobject);
  return bit + newline;
}
function DataObjectToCSV(outputstreams) {
  // csv = getCSVmarkers("list_timing");
  // csv += getCSVheadings(outputstreams);
  csv = getCSVheadings(outputstreams);
  csv += getCSVdata(outputstreams);
  return csv;
}

let dataBuffer = [];
// let windowSize = 2 * 60 * 1000; // 2 minutes in milliseconds
let windowSize = 2 * 60 * samplerate; // 2 minutes in ticks
// let slidingShift = 1000; // 1 second in milliseconds
let slidingShift = samplerate; // 1 second in ticks
let apiUrl = "/predict";

function printHeartRate(event) {
  var dataTime = Date.now();
  var devicename = event.currentTarget.service.device.name;
  var data = event.target.value.buffer;
  var devicetime = Number(new BigInt64Array(data.slice(1, 9))[0]) / 1000000;
  updatecombined[devicename] = true;
  var DataType = Number(new Uint8Array(data.slice(0, 1)));
  console.log(devicename);
  if (!(devicename in intime)) {
    intime[devicename] = devicetime;
  }
  if (graph_time[devicename] > intime[devicename] + duration) {
    intime[devicename] = graph_time[devicename];
    xscale = {
      range: [intime[devicename] - 200, intime[devicename] + duration],
      showticklabels: false,
    }; //ECG graph time (x-axis) range
    layoutCombined[devicename].xaxis = { ...xscale }; //layoutACC.xaxis=xscale;
  }
  //ECG
  if (DataType == 0) {
    //ECG
    samples = new Uint8Array(data.slice(10));
    npoints = samples.byteLength / 3;
    ECGdata = createArray(npoints);
    for (offset = 0; offset < samples.byteLength; offset += 3) {
      i = offset / 3;
      ECGdata[i] = WordstoSignedInteger(samples.slice(offset, offset + 2), 8);
    }
    ECGtime = fillTimeArray(
      DataType,
      devicename,
      data.slice(1, 9),
      dataTime,
      npoints,
      ecg_timestep,
    );
    // flag
    pushData(devicename, ["ECG"], ECGtime, [ECGdata]);
    resolution =
      (layoutCombined[devicename].yaxis.range[1] -
        layoutCombined[devicename].yaxis.range[0]) /
      (document.getElementById("Combined").clientHeight - 100);
    Reduce(devicename, dataobject, "ECG", "ECG_reduced", resolution);
    detectQRS(dataobject[devicename], devicename);
  }
  //PPG
  if (DataType == 1) {
    //PPG
    PPGsum = [];
    PPGsumfilter = [];
    if (!(devicename in last_incom)) {
      last_incom[devicename] = 0;
      last_filter[devicename] = 0;
    }
    NewPPG = completeDeltaFrame(data, 4, 3); //4 channels
    npoints = NewPPG[0].length;
    for (i = 0; i < npoints; i++) {
      PPGsum[i] = NewPPG[0][i] + NewPPG[1][i] + NewPPG[2][i];
    }
    PPGtime = fillTimeArray(
      DataType,
      devicename,
      data.slice(1, 9),
      dataTime,
      npoints,
      1000 / samplerate,
    );
    PPGsumfilter = HPfilter(
      PPGsum,
      last_incom[devicename],
      last_filter[devicename],
    );
    last_filter[devicename] = PPGsumfilter.slice(-1)[0];
    last_incom[devicename] = PPGsum.slice(-1)[0];
    // let last_old_ppgtime = dataObject[devicename]["PPGsum"].x.slice(-1)[0];
    pushData(devicename, ["PPGsum"], PPGtime, [PPGsumfilter]);
    // let new_ppgtime = dataObject[devicename]["PPGsum"].x.slice(-1)[0];

    // log time interval between printHeartRate() calls
    // console.log("time interval between printHeartRate() calls", new_ppgtime - last_old_ppgtime);

    //log data
    // console.log(data);

    smoothPPG(
      devicename,
      dataobject,
      [0.025, 0.04, 0.07, 0.13, 0.2, 0.22, 0.2, 0.13, 0.07, 0.04, 0.025],
    );
    resolution =
      (layoutCombined[devicename].yaxis.range[1] -
        layoutCombined[devicename].yaxis.range[0]) /
      300;
    // document.getElementById("Combined").clientHeight;
    ReduceOld(devicename, dataobject, "PPGsum_f", "PPG_f_reduced", resolution);
    ratePPG(devicename, dataobject);
    ReduceOld(devicename, dataobject, "PPGrate", "PPGrate_reduced", resolution);
    // detectPPGpeak(devicename, dataobject);
  }
  graph_time[devicename] = devicetime; //-firstTime;
}
// run initChart() at inital page load
var option;
var ecgData = [];
document.addEventListener("DOMContentLoaded", initChart);

function initChart() {
  var myChart = echarts.init(document.getElementById("ecgplotchart"));
  option = {
    xAxis: {
      type: "category",
    },
    yAxis: {
      type: "value",
      name: "mV",
    },
    series: [
      {
        name: "ECG",
        // data: ppgsum_f
        data: [],
        type: "line",
        symbol: "none",
      },
      // {
      //   type: "scatter",
      //   data: [],
      //   symbol: "pin",
      //   symbolSize: 30,
      //   label: {
      //     show: true,
      //     formatter: function (d) {
      //       return "{{ label[i+1] }}";
      //     },
      //     position: "inside",
      //     fontWeight: "bold",
      //   },
      // },
    ],
    grid: {
      left: 30,
      top: 10,
      right: 20,
      bottom: 30,
    },
    dataZoom: [
      {
        type: "inside",
        start: 70,
        end: 100,
      },
      {
        start: 0,
        end: 10,
      },
    ],
  };

  myChart.setOption(option);
  //responsive widht
  window.addEventListener("resize", function () {
    myChart.resize();
  });

  // // refresh at 1 second
  // if (dataobject[name] !== undefined) {
  //   setInterval(function () {
  //     myChart.setOption({
  //       series: [
  //         {
  //           name: "ECG",
  //           data: dataobject[name].PPGsum_f.y,
  //         },
  //       ],
  //     });
  //   }, 1000);
  // }
}

// function AddMarker() {
//   marker = document.getElementById("markername");
//   markervalue = marker.value;
//   if (markervalue != "") {
//     text =
//       "<button title='Insert marker: " +
//       markervalue +
//       "' onClick='InsertMarker(\"" +
//       markervalue +
//       "\")'>" +
//       markervalue +
//       "</button>";
//     document.getElementById("Markers").insertAdjacentHTML("beforeend", text);
//   }
//   marker.focus();
//   marker.value = "";
// }
// function InsertMarker(a) {
//   var entrytime = new Date();
//   var ISOtime = entrytime.toISOString();
//   entrytime = (entrytime / 10) * 10; ///24/60/60+25569;
//   document.getElementById("list_timing").innerHTML =
//     document.getElementById("list_timing").innerHTML +
//     "<tr><td>" +
//     ISOtime +
//     "</td><td>" +
//     entrytime +
//     "</td><td>" +
//     a +
//     "</td></tr>";
// }

// function updateGraphs() {
//   //lines=[];
//   for (const [key, trace] of Object.entries(traces)) {
//     graphname = "graph" + key;
//     if (
//       updatecombined[key] == true &&
//       document.getElementById(graphname) != null
//     ) {
//       layoutCombined[key].datarevision = Math.random();
//       linestoplot = [
//         "PPGpeak",
//         "PPGstart",
//         "PPGend",
//         "PPGrate_reduced",
//         "PPG_f_reduced",
//         "acc_v_reduced",
//         "ECG_reduced",
//         "QRS",
//       ];
//       lines = [];
//       for (id in trace) {
//         if (linestoplot.includes(id)) {
//           lines.push(trace[id]);
//         }
//       }
//       //console.log(graphname, lines, layoutCombined[key]);
//       Plotly.react(graphname, lines, layoutCombined[key]);
//       updatecombined[key] = false;
//     }
//   }
//   //Plotly.react(graphname,lines,layoutCombined[key]);
// }