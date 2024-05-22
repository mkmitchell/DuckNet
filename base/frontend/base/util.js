
function deepcopy(x){
    return JSON.parse(JSON.stringify(x))
}

function sleep(ms){  //XXX: await sleep(x)
    return new Promise(resolve => setTimeout(resolve, ms));
}

function argmin(x){
    return arange(x.length).reduce( (carry,i) => x[i]<x[carry]? i : carry );
}

function arange(x0,x1=undefined){
    var start = (x1==undefined)?  0 : x0;
    var stop  = (x1==undefined)? x0 : x1-start;
    return [...Array(stop).keys()].map(x=>x+start)
}

function remove_file_extension(filename){
    return filename.slice(0, filename.lastIndexOf('.'))
}

function file_basename(filename){
    return filename.slice(filename.lastIndexOf('/')+1)
}

function rename_file(file, newname){
    return new File([file], newname, {type: file.type});
}

function upload_file_to_flask(file, url='file_upload', async=true){
    var formdata = new FormData();
    formdata.append('files', file);

    return $.post({
        url:         url,
        data:        formdata,
        processData: false,
        cache:       false,
        contentType: false,
        async:       async,
        enctype:     'multipart/form-data'
    })
}

function url_for_image(imagename, cachebuster=true){
    return `/images/${imagename}` + (cachebuster? `?_=${Date.now()}` : '')
}

//fetch request that returns a blob
async function fetch_as_blob(url){
    const r = await fetch(url);
    return await (r.ok ? r.blob() : undefined);
}

//fetch request that returns a file
async function fetch_as_file(url){
    var filename = url.split('/').reverse()[0].split('?')[0]
    const b      = await fetch_as_blob(url);
    return new File([b], filename, { type: b.type });
}

//downloads an element from the uri (to the user hard drive)
function downloadURI(filename, uri) {
    var element = document.createElement('a');
    element.setAttribute('href', uri);
    element.setAttribute('download', filename);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}
  
function download_text(filename, text){
    return downloadURI(filename, 'data:text/plain;charset=utf-8,'+encodeURIComponent(text))
}
  
function download_blob(filename, blob){
    return downloadURI(filename, URL.createObjectURL(blob));
}

async function download_zip(filename, zipdata){
    var zip = new JSZip();
    for(var fname in zipdata)
        zip.file(fname, await zipdata[fname], {binary:true});
    
    zip.generateAsync({type:"blob"}).then( blob => {
        download_blob(filename, blob  );
    } );
}



function is_string(x){
    return (x instanceof String || typeof x === "string")
}

function sort_object(o, sorted_keys){
    return sorted_keys.reduce( (new_o, k) => (new_o[k] = o[k], new_o), {} );
}

function sort_object_by_value(o) {
    return Object.keys(o).sort(function(a,b){return o[b]-o[a]}).reduce((r, k) => (r[k] = o[k], r), {});
}


//parses a string like "matrix(1,0,0,1,0,0)"  //TODO: use DOMMatrix() instead
function parse_css_matrix(maxtrix_str){
    var x      = Number(maxtrix_str.split(')')[0].split(', ')[4])
    var y      = Number(maxtrix_str.split(')')[0].split(', ')[5])
    var scale  = Number(maxtrix_str.split('(')[1].split(',')[0])
    return {x:x, y:y, scale:scale};
}


//reload/update a javascript file (for development)
function reload_script(url){
    //first get index.html so that flask recompiles the static folder
    //then reload all app.js files
    //FIXME: too complicated
    var app_scripts    = $('script[src*="app.js"]').get().map( x => x.src )
    var url_scripts    = $(`script[src*="${url}"]`).get().map( x => x.src )
    var scripts_to_reload = [].concat(url_scripts).concat(app_scripts)
    $.get('/').then(
        _ => scripts_to_reload.reduce( async (prev, u) => {await prev; return $.getScript(`${u}`)}, 'init' )
    )
}
