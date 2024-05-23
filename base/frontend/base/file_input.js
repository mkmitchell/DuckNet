

BaseFileInput = class {
    //called when user selects input file(s)
    static on_inputfiles_select(event){
        //this.set_input_files(event.target.files);
        this.load_list_of_files(event.target.files)  //not using set_input_files for tests
        event.target.value = ""; //reset the input
    }
    
    //called when user selects an input folder
    static on_inputfolder_select(event){
        //no using load_list_of_files() to avoid ambiguity
        var files = [];
        for(var f of event.target.files)
            if(f.type.startsWith('image'))
                files.push(f);
        this.set_input_files(files);
        event.target.value = ""; //reset the input
    }

    static on_annotations_select(event){
        this.load_result_files(event.target.files)
        event.target.value = ""; //reset the input
    }

    static set_input_files(files){
        //send request to flask to clear cache folder
        if(!window.location.href.startsWith('file://'))
            $.get('/clear_cache')

        GLOBAL.files = []
        for(let f of files)
            GLOBAL.files[f.name] = new InputFile(f)
        //FIXME: currently the detection tab needs to be visible
        $('.tabs .item[data-tab="detection"]').click()
        return this.refresh_filetable(files)
    }

    //update the ui accordion table
    static async refresh_filetable(files){
        var $filetable = $('#filetable');
        if(!$filetable.is(':visible'))
            console.error('Detection file table is not visible')
        $filetable.find('tbody').html('');

        //show progress
        const $modal = $('#loading-files-modal')
        $modal.modal({closable: false, inverted:true, duration : 0,}).modal('show');
        $modal.find('.progress').progress({
            total: files.length,
            value: 0, showActivity:false,
        })

        const insert_single_table_row = async function(i, resolve){
            const f     = files[i]
            if(!f){
                //FIXME: ugly/hacky (but faster than using <script> in every row)
                const $after_inserts = $('after-insert-script')
                const scripts = [...(new Set($after_inserts.get().map(x => x.innerHTML.trim()) ))]
                scripts.map(eval)
                //$after_inserts.remove()   //slow if many
                
                $modal.modal({closable: true}).modal('hide');
                //await sleep(500)    //XXX? needed?
                $modal.find('.progress').progress('reset')

                resolve()
                return;
            }
            
            const $trow = $("template#filetable-row-template").tmpl([{filename:f.name}])
            $trow.appendTo($filetable.find('tbody'));
            //get the y-coordinate of the row, as long as all rows are closed
            //would be unreliable later on
            //FIXME: works only if visible
            $trow.first().attr('top', $trow.offset().top)

            $filetable.find('thead th#files-loaded-column-header').text(`${i+1} File${(i+1==1)?'':'s'} Loaded`)
            $modal.find('.progress').progress('set progress', i)

            //using timeouts to avoid frozen UI
            setTimeout(() => {
                insert_single_table_row(i+1, resolve);
            }, 0);
        }

        return new Promise((resolve, _reject) => {
            insert_single_table_row(0, resolve)
        });
    }


    static setup_drag_and_drop(){
        $('body').on('dragover', e => this.on_dragover(e.originalEvent))
        $('body').on('drop',     e => this.on_drop(e.originalEvent))
    }
    
    static on_dragover(event){
        event.preventDefault();
    }

    static on_drop(event){
        event.preventDefault();
        console.log('on_drop:', event)
        this.load_list_of_files(event.dataTransfer.files)
    }

    //load files, some might be input files, others results
    static async load_list_of_files(files){
        var files = Array(...files)
        var inputfiles = files.filter( f => ["image/jpeg", "image/tiff"].indexOf(f.type)!=-1); //no png
        if(inputfiles.length)
            await this.set_input_files(inputfiles)
        
        var remaining_files = files.filter( f => inputfiles.indexOf(f)==-1 )
        this.load_result_files(remaining_files)
    }

    //load files as results if the match already loaded input files
    static async load_result_files(files){
        console.warn('load_result_files', performance.now())
        var result_files = await this.collect_result_files(files)
        console.log('result_files.length:', Object.keys(result_files).length)
        if(Object.keys(result_files).length==0)
            return;
        
        //show progress
        var $modal = $('#loading-files-modal')
        $modal.modal({closable: false, inverted:true, duration : 0,}).modal('show');
        $modal.find('.progress').progress({
            total: Object.keys(result_files).length,
            value: 0, showActivity:false,
        })
        try{
            for(const [filename, results] of Object.entries(result_files)){
                const unzipped_results = await Promise.all(results.map(maybe_unzip))
                await this.load_result(filename, unzipped_results )
                $modal.find('.progress').progress('increment')
            }
        } catch(error) {
            console.error(error);
            $('body').toast({message:'Failed loading results.', class:'error', displayTime: 0, closeIcon: true})
        } finally {
            $modal.modal({closable: true}).modal('hide');
            await sleep(500)
            $modal.find('.progress').progress('reset')
        }
    }

    //filter a list of files, leaving result files for already loaded input files
    static async collect_result_files(files){
        var collected = {}
        for(var f of Object.values(files)){
            if(["application/zip", "application/x-zip-compressed"].indexOf(f.type)!=-1)
                Object.assign(collected, (await this.collect_results_from_zipfile(f)));
            else
                for(var filename of Object.keys(GLOBAL.files))
                    if( this.match_resultfile_to_inputfile(filename, f.name) ){
                        const prev = collected.hasOwnProperty(filename) ? collected[filename] : []
                        collected[filename] = prev.concat([f])
                    }
        }
        return collected;
    }

    static async collect_results_from_zipfile(zipfile){
        var zipped_files = (await JSZip.loadAsync(zipfile)).files
        return await this.collect_result_files(zipped_files)
    }

    //return true if the result file matches the input file
    static match_resultfile_to_inputfile(inputfilename, resultfilename){
        var basename          = file_basename(resultfilename)
        const no_ext_filename = remove_file_extension(inputfilename)
        const candidate_names = [
            inputfilename+'.png',   inputfilename+'.segmentation.png', 
            no_ext_filename+'.png', no_ext_filename+'.segmentation.png'
        ]
        return (candidate_names.indexOf(basename) != -1)
    }

    //load a single result file for an input image
    //overwritten downstream
    static async load_result(filename, resultfiles){
        const resultfile = resultfiles[0]
        const blob   = await(resultfile.async? resultfile.async('blob') : resultfile)
        const file   = new File([blob], resultfile.name, {type:'image/png'})
        const result = {segmentation: file}
        GLOBAL.App.Detection.set_results(filename, result)
    }
};


async function maybe_unzip(file, default_type='image/png'){
    if(file.async != undefined){
        const blob = await(file.async? file.async('blob') : file)
        const type = file.type? file.type : default_type;
              file = new File([blob], file.name, {type:type})
    }
    return file;
}
