
BaseDetection = class {

    //process single image
    static on_process_image(event){
        var filename = $(event.target).closest('[filename]').attr('filename')
        this.process_image(filename)
    }

    //process all images
    static async on_process_all(event){
        var filenames = Object.keys(GLOBAL.files)
        console.log(`Processing ${filenames.length} images.`)
        //TODO: disable stuff: settings, loading files, etc
        //TODO: show processing progress
        $(event.target).hide()
        $(event.target).siblings('.cancel-processing, .processing').show().removeClass('disabled')
        GLOBAL.cancel_requested = false;
        for(var filename of filenames){
            if(GLOBAL.cancel_requested)
                break;
            
            try {
                await this.process_image(filename)  //TODO: error handling
            } catch {
                $('body').toast({message:'Processing failed.', class:'error', displayTime: 0, closeIcon: true})
                break;
            }
        }
        $(event.target).show()
        $(event.target).siblings('.cancel-processing, .processing').hide()
    }

    static on_cancel_processing(event){
        GLOBAL.cancel_requested = true;
        $(event.target).hide()
        $(event.target).siblings('.processing').addClass('disabled')
    }

    //TODO: convert to async function?
    static process_image(filename){
        console.log(`Processing image file ${filename}`)
        //this.clear_results(filename)
        this.set_results(filename, undefined)
        this.set_processing(filename)
        
        //Called on a server-side event from the server notifying about processing progress
        function on_message(event){
            var data = JSON.parse(event.originalEvent.data);
            if(data.image!=filename)
                return;

            console.log(event)
            //TODO: update dimmer
        }
        $(GLOBAL.event_source).on('message', on_message)

        var file    = GLOBAL.files[filename];
        let promise = upload_file_to_flask(file)
        promise.fail( response => {
            console.log('File upload failed.', response.status)
            this.set_failed(filename)
        })

        const _this = this;
        promise = promise.then( function(){
            return $.get(`process_image/${filename}`).fail( response => {
                console.log('Processing failed.', response.status)
                _this.set_failed(filename)
            })
        } )
        promise.done(results => this.set_results(filename, results))


        promise.always( _ => {
            //TODO: delete_image_from_flask(filename)
            $(GLOBAL.event_source).off('message', on_message)
        })
        return promise;
    }

    static async set_results(filename, results){
        //TODO: lazy loading
        var clear = (results == undefined)
        this.hide_dimmer(filename)

        if(results && is_string(results.segmentation))
            results.segmentation = await fetch_as_file(url_for_image(results.segmentation))
        var segmentation = clear? undefined : results.segmentation;

        var $root           = $(`#filetable [filename="${filename}"]`)
        //var $result_image   = $root.find('img.result-image')
        //GLOBAL.App.ImageLoading.set_image_src($result_image, segmentation)
        //$result_image.css('filter',clear? 'contrast(0)' : 'contrast(1)')
        $root.find('.show-results-checkbox')
            .checkbox({onChange: () => GLOBAL.App.ViewControls.toggle_results(filename)})
            .checkbox('check')

        var $result_overlay = $root.find(`img.overlay`)
        GLOBAL.App.ImageLoading.set_image_src($result_overlay, segmentation)

        GLOBAL.files[filename].set_results(results)
        GLOBAL.App.Boxes.refresh_boxes(filename)

        this.set_processed(filename, clear)
    }

    static set_processing(filename){
        this.show_dimmer(filename, false)
        var $root      = $(`#filetable [filename="${filename}"]`)

        $(`.table-row[filename="${filename}"] label`).css('font-weight', '')
        $root.find('.status.icon').hide().filter('.spinner').show()
        
        this.enable_buttons(filename, false, false)
    }

    static set_failed(filename){
        this.show_dimmer(filename, true)
        var $root      = $(`#filetable [filename="${filename}"]`)

        $(`.table-row[filename="${filename}"] label`).css('font-weight', '')
        $root.find('.status.icon').hide().filter('.exclamation').show()

        this.enable_buttons(filename, true, false)
    }

    static set_processed(filename, clear=false){
        this.hide_dimmer(filename)
        this.enable_buttons(filename, true, !clear)
        //indicate in the file table that this file is processed
        $(`.table-row[filename="${filename}"] label`).css('font-weight', clear? '' : 'bold')
        $(`#filetable [filename="${filename}"] .status.icon`).hide().filter(clear? '.unprocessed' : '.processed').show()
    }

    static hide_dimmer(filename){
        $(`[filename="${filename}"] .dimmer`).dimmer('hide')
    }

    static show_dimmer(filename, failed, message='Processing...'){
        var $dimmer = $(`[filename="${filename}"] .dimmer`)
        if(failed){
            $dimmer.find('.content.processing').hide()
            $dimmer.find('.content.failed').show()
            $dimmer.dimmer({closable: true})
        } else {
            $dimmer.find('.content.processing').show()
            $dimmer.find('.content.failed').hide()
            $dimmer.dimmer({closable: false})
            $dimmer.find('.processing p').text(message)
        }
        $dimmer.dimmer('show')
    }

    static enable_buttons(filename, yesno=true, download=true){
        //TODO: also disable settings, 'Process All' button, etc
        $(`[filename="${filename}"] .icon.menu .item`)
            .toggleClass('disabled', !yesno)
            .filter('.download').toggleClass('disabled', !download)
    }

}