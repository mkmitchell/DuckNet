

BaseBoxes = class {
    static on_draw_new_box_button(event){
        const $root    = $(event.target).closest('[filename]')
        const filename = $root.attr('filename')
        const active   = $root.find('.item.new-box').hasClass('active');
        this.set_box_drawing_mode(filename, !active);
    }

    static set_box_drawing_mode(filename, state){
        const $root = $(`[filename="${filename}"]`)
        $root.find(`.item.new-box`).toggleClass('active', state)
        
        const $container  = $root.find('.transform-box')
        $container.css('cursor', state? 'crosshair' : '')

        if(!state)
            return;
        
        //temporary box overlay
        const $selection = $('<div>').css({"background": "transparent", 
                                           "position":   "absolute", 
                                           "border":     "1px dotted #fff"});
        const _this = this;
        $container.one('mousedown', function(mousedown_event) {
            const img                 = $container.find('img.input-image')[0]
            const [click_x, click_y]  = page2img_coordinates([mousedown_event.pageX, mousedown_event.pageY], img, $container)
    
            $selection.css({
              'top':    click_y,  'left':   click_x,
              'width':  0,        'height': 0
            });
            $selection.appendTo($container);

            $(document).on('mousemove mouseup', function(mousemove_event) {
                if( (mousemove_event.buttons & 0x01)==0 ){
                    //mouse up
                    
                    $(document).off('mousemove mouseup');
                    _this.set_box_drawing_mode(filename, false);

                    _this.finalize_box($selection, '???')
                    $selection.remove()
                    return;
                }

                const [move_x, move_y] = page2img_coordinates([mousemove_event.pageX, mousemove_event.pageY], img, $container)
                const width  = Math.abs(move_x - click_x),
                      height = Math.abs(move_y - click_y);
    
                const new_x  = (move_x < click_x) ? (click_x - width)  : click_x;
                const new_y  = (move_y < click_y) ? (click_y - height) : click_y;
    
                $selection.css({
                  'width': width,  'height': height,
                  'top':   new_y,  'left':   new_x
                });
            })
        })
    }

    static finalize_box( $box_overlay, label, remove=false ){
        const filename      = $box_overlay.closest('[filename]').attr('filename')
        
        const $image = $(`[filename="${filename}"] img.input-image`)
        const rect   = $image[0].getBoundingClientRect()
        const [H,W]  = [rect['height'], rect['width']]
        //normalized coordinates 0..1
        const box_position = $box_overlay.position()
        const box_rect     = $box_overlay[0].getBoundingClientRect()
        const y0  = Math.max(0, box_position['top']/H)
        const x0  = Math.max(0, box_position['left']/W)
        const y1  = Math.min(1, (box_position['top']  + box_rect['height'])/H );
        const x1  = Math.min(1, (box_position['left'] + box_rect['width']) /W );

        const [imgW,imgH] = GLOBAL.App.ImageLoading.get_imagesize(filename)
        //real image coordinates
        const box   = [x0*imgW, y0*imgH, x1*imgW, y1*imgH]

        //update results
        const oldresults = GLOBAL.files[filename].results;
        let   newresults = Object.assign(deepcopy(oldresults), {
            'labels': oldresults?.['predictions']  ?? [],
            'boxes':  oldresults?.['boxes']        ?? [],
        })
        const index = $box_overlay.attr('index')
        if(remove){
            //remove box
            newresults['labels'].splice(index, 1)
            newresults['boxes'].splice(index, 1)
        } else if(index == undefined){
            //add new box
            newresults['labels'] = newresults['labels'].concat({[label]:1.0})
            newresults['boxes']  = newresults['boxes'].concat([box])
        } else {
            //update old result at index
            if(label != undefined)
                newresults['labels'][index] = {[label]:1.0};
            newresults['boxes'][index]  = box;
        }
        GLOBAL.App.Detection.set_results(filename, newresults)
    }

    static add_box_overlay(filename, yxyx, label, index){
        const display_label = (!!label)? label : GLOBAL.App.NEGATIVE_CLASS;
        const $overlay   = $('#box-overlay-template').tmpl({box:yxyx, label:display_label, index:index})
        const $container = $(`[filename="${filename}"] .boxes.overlay`)
        $overlay.appendTo($container)
        const tooltip_text = this.tooltip_text(filename, index)
        if(tooltip_text)
            $overlay.find('.box-label-container').popup({'html':tooltip_text});
        
        //clip label position to stay within image borders
        const box_label = $overlay.find('.box-label-container')[0]
        const top = ($container[0].getBoundingClientRect().top - box_label.getBoundingClientRect().top - 20)
        $(box_label).css('top', Math.max(top, -20) )


        const _this = this;
        function stop_drag(){
            $(document).off('mousemove mouseup');

            _this.finalize_box($overlay, undefined)
            return;
        }

        const $xformbox = $(`[filename="${filename}"] .transform-box`)
        const img       = $xformbox.find('img.input-image')[0]
        $overlay.find('.move-anchor').on('mousedown', function(mousedown_event){
            const [click_x, click_y]  = page2img_coordinates([mousedown_event.pageX, mousedown_event.pageY], img, $xformbox)
            const overlay_y = $overlay[0].offsetTop;
            const overlay_x = $overlay[0].offsetLeft;
            //make sure height/width are fixed
            $overlay.css('height', $overlay.css('height'));
            $overlay.css('width',  $overlay.css('width'));
    
            $(document).on('mousemove mouseup', function(mousemove_event) {
                if( (mousemove_event.buttons & 0x01)==0 ){
                    stop_drag();
                    return;
                }

                const [move_x, move_y] = page2img_coordinates([mousemove_event.pageX, mousemove_event.pageY], img, $xformbox)
                $overlay.css({
                    'top':  overlay_y + (move_y - click_y), 
                    'left': overlay_x + (move_x - click_x), 
                  });
            });
            return false; //stop event propagation
        })
    
        $overlay.find('.resize-anchor').on('mousedown', function(mousedown_event){
            const [click_x, click_y]  = page2img_coordinates([mousedown_event.pageX, mousedown_event.pageY], img, $xformbox)
            const overlay_h = $overlay.outerHeight();
            const overlay_w = $overlay.outerWidth();
    
            $(document).on('mousemove mouseup', function(mousemove_event) {
                if( (mousemove_event.buttons & 0x01)==0 ){
                    stop_drag();
                    return;
                }

                const [move_x, move_y] = page2img_coordinates([mousemove_event.pageX, mousemove_event.pageY], img, $xformbox)
                $overlay.css({
                    'height': overlay_h + (move_y - click_y), 
                    'width':  overlay_w + (move_x - click_x), 
                  });
            });
            return false; //stop event propagation
        })
    }

    static tooltip_text(filename, index){
        const results    = GLOBAL.files[filename].results;
        if( !results.predictions )
            return '';
        
        const prediction = results.predictions[index]
        if(Object.keys(prediction).length==0)
            return ''
        
        let txt = '<b>Prediction:</b>';
        for(const [label, confidence] of Object.entries(prediction)){
            if(confidence < 0.1)
                continue;
            txt += `<br/>${label? label:GLOBAL.App.NEGATIVE_CLASS}: ${ (prediction[label]*100).toFixed(0) }%`
        }
        return txt;
    }

    static refresh_boxes(filename){
        const img = $(`[filename="${filename}"] img.input-image`)[0];
        if(img.naturalWidth==0){
            //image not yet loaded, display on load
            $(img).one( 'load', _ => this.refresh_boxes(filename) )
            return;
        }
        const [W,H] = GLOBAL.App.ImageLoading.get_imagesize(filename)
        this.clear_box_overlays(filename)

        const results = GLOBAL.files[filename]?.results;
        const boxes   = results?.boxes;
        const labels  = results?.labels;
        if(!boxes || !labels)
            return;
        
        for(const [i,box] of Object.entries(boxes)){
            const yxyx = [box[1]/H, box[0]/W, box[3]/H, box[2]/W]
            this.add_box_overlay(filename, yxyx, labels[i], i)
        }
    }

    static on_remove_box_button(event){
        const $box_overlay = $(event.target).closest('.box-overlay')
        this.finalize_box($box_overlay, undefined, true)
    }

    static clear_box_overlays(filename){
        const $container = $(`[filename="${filename}"] .boxes.overlay`)
        $container.find('.box-overlay').remove()
    }


    static convert_boxlabel_into_input(event) {
        const $label = $(event.target)
        //activate dropdown
        $label.closest('.box-overlay').find('select.search').dropdown({
            allowAdditions: true, 
            hideAdditions:  false, 
            forceSelection: false, 
            selectOnKeydown: false,
            fullTextSearch:  true,
            silent:          true,
            action: (t,v,el) => {  save(t); },
            onHide: ()       => {  save( ); },
        });
        const $input = $label.closest('.box-overlay').find('.search.dropdown');
        $input.dropdown('setup menu', {
            values: this.get_set_of_all_labels().map( v => {return {name:v};} ),
        });
        //$label.hide();
        $label.css('visibility', 'hidden');
        $input.show();
    
        const _this = this;
        const save = function(txt=''){
            //console.log(`save(${txt})`)
            if(txt.length > 0){
                $label.text(txt)
                
                const box_overlay = $label.closest('.box-overlay')
                $input.dropdown('unbind intent')    //keep this; seems to prevent an error message
                _this.finalize_box(box_overlay, txt);
            }
            $label.show().css('visibility', '');
            $input.hide();
        }
        $input.find('input').focus().select();
    }

    static get_set_of_all_labels(){
        //to be implemented downstream
        return []
    }
}



//translate page coordinates xy to img coordinates
//(viewport element provides topleft corner and transform)
function page2img_coordinates(page_xy, img, $viewport){
    let H,W;
    if(navigator.userAgent.indexOf('Chrom') != -1){
        //some layout issues with chrome
        H = img.clientHeight;  //integer
        W = img.clientWidth;
    } else {
        H = $(img).height()      //float
        W = $(img).width()
    }
    const xform        = parse_css_matrix($viewport.css('transform'));
    //absolute coordinates on the html element in pixels
    const html_x_abs   = page_xy[0] - $viewport.offset().left
    const html_y_abs   = page_xy[1] - $viewport.offset().top
    //relative coordinates on the html element, range 0.0..1.0
    const html_x_rel   = html_x_abs / W / xform.scale
    const html_y_rel   = html_y_abs / H / xform.scale
    //absolute coordinates on the image element
    //const img_x_abs    = html_x_rel * img.naturalWidth
    //const img_y_abs    = html_y_rel * img.naturalHeight
    const img_x_abs    = html_x_rel * img.width
    const img_y_abs    = html_y_rel * img.height
    
    return [img_x_abs, img_y_abs];
}
