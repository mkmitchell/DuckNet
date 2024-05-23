
BaseImageLoading = class {
    static on_accordion_open(table_row){
        var $root    = $(table_row).closest('[filename]')
        var filename = $root.attr('filename')
        var file     = GLOBAL.files[filename];
        var $img     = $root.find('img.input-image')
    
        if(this.is_image_loaded($img)){
            this.scroll_to_filename(filename)  //won't work the first time
            return;
        }
        $img.on('load', _ => this.rescale_image_if_too_large($img[0])) //TODO: also rescale result images
        $img.one('load', () => {
            var $par = $root.find('.set-aspect-ratio-manually')
            var img  = $img[0]
            $par.css('--imagewidth',  img.naturalWidth)
            $par.css('--imageheight', img.naturalHeight)
    
            $root.find('.loading-message').remove()
            $root.find('.filetable-content').show()
            this.scroll_to_filename(filename)  //works on the first time
        })
        this.set_image_src($img, file);
    
        //setting the result image as well, only to get the same dimensions
        //if not already loaded from results
        var $result_img = $root.find('img.result-image')
        if($result_img.length && !this.is_image_loaded($result_img))
            this.set_image_src($result_img, file);  //TODO: generate new dummy image with same aspect ratio
    }
    
    
    static set_image_src($img, file){
        if( (file instanceof File)
         && (file.type=="image/tiff" || file.name.endsWith('.tif') || file.name.endsWith('.tiff'))) {
            this.load_tiff_file(file).then( blob => this.set_image_src($img, blob) )
        } else if( file instanceof Blob) {
            var url = URL.createObjectURL(file)
            $img.attr('src', url).css('visibility', '')
            $img.one('load', _ => URL.revokeObjectURL(url) )
            console.log('Setting image src of', $img, 'to blob', file)
        } else if (is_string(file)){
            var url = url_for_image(file)
            $img.attr('src', url).css('visibility', '')
            console.log('Setting image src of', $img, 'to string', file)
        } else if (file==undefined) {
            //hidden to prevent the browser showing a placeholder
            $img.css('visibility', 'hidden')
            $img[0].removeAttribute('src')
        } else {
            throw TypeError(`Cannot set image src to ${file}`)
        }
    }
    
    static is_image_loaded($img){
        return ($img.attr('src')!=undefined)
    }
    
    static scroll_to_filename(filename){
        var $root       = $(`tr.ui.title[filename="${filename}"]`)
        var top         = $root.attr('top')
        setTimeout(() => {
            window.scrollTo( {top:top, behavior:'smooth'} )
        }, 10);
    }
    
    static async load_tiff_pages(file){
        const promise = new Promise( (resolve, reject) => {
            const freader = new FileReader()
            freader.onload = function(event){
                const buffer = event.target.result
                const pages  = UTIF.decode(buffer);
                resolve([pages, buffer])
            }
            freader.readAsArrayBuffer(file);
        } )
        return promise;
    }
    
    static async load_tiff_file(file, page_nr = 0){
        const [pages, buffer]   = await this.load_tiff_pages(file)
        const promise = new Promise( (resolve, reject) => {
            const page   = pages[page_nr]
            UTIF.decodeImage(buffer, page, pages)
            const rgba   = UTIF.toRGBA8(page);
            const canvas = $(`<canvas width="${page.width}" height="${page.height}">`)[0]
            const ctx    = canvas.getContext('2d')
            ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer), page.width, page.height), 0, 0);
            canvas.toBlob(blob =>  resolve(blob), 'image/jpeg', 0.92);
        });
        return promise
    }
    
    
    static rescale_image_if_too_large(img, acceptable_area=2048**2){
        const area   = img.naturalHeight * img.naturalWidth;
        const ratio  = acceptable_area**0.5 / area**0.5
        const smooth = img.classList.contains('input-image')
        if(ratio < 0.95)  //1.0 is prone to infinite loop
            this.rescale_image(img, ratio, smooth)
    }
    
    static rescale_image(img, scale=0.5, smoothing=false){
        const t0     = Date.now()
        const [W,H]  = [img.naturalWidth, img.naturalHeight];
        const [w,h]  = [Math.round(W*scale), Math.round(H*scale)]
        console.log(`Rescaling image from [${W}x${H}] to [${w}x${h}] and smoothing: ${smoothing}`)
        const canvas = $(`<canvas width="${w}" height="${h}">`)[0]
        const ctx    = canvas.getContext('2d')
        ctx.imageSmoothingEnabled = smoothing;
        ctx.drawImage(img, 0,0, w,h)
        const format = smoothing? 'image/jpeg' : 'image/png'
        $(img).one('load', _ => console.log(`Image rescaled in ${Date.now() - t0}ms`) )
        canvas.toBlob(blob =>  this.set_image_src($(img), blob), format, 0.92 );
    }
    
    //return the real/natural image size (or undefined if image not yet loaded)
    static get_imagesize(filename){
        const $container = $(`[filename="${filename}"] .set-aspect-ratio-manually`)
        const img        = $(`[filename="${filename}"] img.input-image`)[0]
        if($container.length){
            const W  = Number($container.css('--imagewidth')  ?? img.naturalWidth  );
            const H  = Number($container.css('--imageheight') ?? img.naturalHeight );
            return [W,H]
        }
        return undefined;
    }
}



