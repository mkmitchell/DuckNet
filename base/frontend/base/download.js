
BaseDownload = class {
    static async on_single_item_download_click(event){
        var $root     = $(event.target).closest('[filename]')
        var filename  = $root.attr('filename')

        var zipdata  = this.zipdata_for_file(filename);
        if(zipdata == undefined){
            $('body').toast({message:'Result download failed.', class:'error'})
            return
        }
        download_zip(`${filename}.results.zip`, zipdata)
    }

    static on_download_all(event){
        var filenames = Object.keys(GLOBAL.files)
        var zipdata   = this.zipdata_for_files(filenames)
        if(Object.keys(zipdata).length==0)
            return
        download_zip('results.zip', zipdata)
    }

    //to be overwritten downstream
    static zipdata_for_file(filename){
        var f            = GLOBAL.files[filename];
        var zipdata      = {};
        var segmentation = f.results.segmentation
        zipdata[`${f.results.segmentation.name}`] = segmentation  //TODO: folders
        return zipdata;
    }

    static zipdata_for_files(filenames){
        var zipdata   = {}
        for(var filename of filenames){
            var fzipdata = this.zipdata_for_file(filename)
            if(fzipdata == undefined)
                continue;
            
            for(var k of Object.keys(fzipdata))
                zipdata[`${filename}/${k}`] = fzipdata[k];
        }
        return zipdata;
    }
}


ObjectDetectionDownload = class extends BaseDownload{
    //callback to download annotation .json files
    static on_download_json(event){
        let zipdata = {}
        for(const [filename, f] of Object.entries(GLOBAL.files)){
            if(!f.results)
                continue;

            const jsonfile         = this.build_annotation_jsonfile(filename, f.results)
            zipdata[jsonfile.name] = jsonfile
        }
        if(Object.keys(zipdata).length > 0)
            download_zip('annotations.zip', zipdata)
    }

    static build_annotation_jsonfile(filename, results){
        let jsondata = deepcopy(LABELME_TEMPLATE);
        jsondata = Object.assign(jsondata, results)
        jsondata.imagePath = filename

        for(const [i,box] of Object.entries(results.boxes ?? {})){
            const label      = results.labels[i].trim() || GLOBAL.App.NEGATIVE_CLASS;
            //if(label.trim()=="")
            //    continue;
            let jsonshape    = deepcopy(LABELME_SHAPE_TEMPLATE);
            jsonshape.label  = label;
            jsonshape.points = [ [box[0], box[1]], [box[2], box[3]] ];
            jsondata.shapes.push(jsonshape);
        }

        const jsonfilename = filename.split('.').slice(0, -1).join('.')+'.json'
        const blob         = new Blob([JSON.stringify(jsondata, null, 2)], {type : 'application/json'})
        const jsonfile     = new File([blob], jsonfilename)
        return jsonfile
    }
}


const LABELME_TEMPLATE = {
    //version: "3.16.2",
    flags: {},
    shapes: [    ],
    lineColor: [ 0, 255, 0, 128 ],
    fillColor: [255,  0, 0, 128 ],
    imagePath: "",
    imageData: null
}

const LABELME_SHAPE_TEMPLATE = {
    label: "???",
    line_color: null,
    fill_color: null,
    points: [ [ 2297.6377952755906, 2039.3700787401574 ],
              [ 3204.7244094488187, 2317.3228346456694 ] ],
    shape_type: "rectangle",
    flags: {}
}


