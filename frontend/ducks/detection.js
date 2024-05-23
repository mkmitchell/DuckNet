DuckDetection = class extends BaseDetection {

    //override
    static async set_results(filename, results){
        const clear = (results == undefined)
        this.hide_dimmer(filename)

        GLOBAL.files[filename].results = undefined;
        GLOBAL.App.Boxes.clear_box_overlays(filename)

        const $root = $(`#filetable [filename="${filename}"]`)
        $root.find('.show-results-checkbox')
            .checkbox({onChange: () => GLOBAL.App.ViewControls.toggle_results(filename)})
            .checkbox('check')
        
        if(!clear){
            console.log(`Setting results for ${filename}:`, results)
            const duckresults = new DuckResults(results)
            GLOBAL.files[filename].results = duckresults
            GLOBAL.App.Boxes.refresh_boxes(filename)
            
            $root.find(`td:nth-of-type(2)`).html( this.format_results_for_table(duckresults) )
            this.update_flags(filename)
        }

        this.set_processed(filename, clear)
    }

    static format_results_for_table(duckresults){
        const hiconf_threshold = GLOBAL.settings.confidence_threshold/100 ?? 0.70
        const n     = duckresults.labels.length;
        console.log('duckresults', duckresults)
        console.log('duckresult labels', duckresults.labels)
        let   texts = []
        for (let i = 0; i < n; i++) {
            let   label      = duckresults.labels[i];
            console.log('label ' +label)
            const confidence = Object.values(duckresults.predictions[i])[0]
            if(!label || (label.toLowerCase()=='not-a-duck')){
                if(confidence > hiconf_threshold)
                    //filter high-confidence non-duck
                    continue;
                else
                    label = 'Not-A-Duck'
            }
            
            let   text       = `${label}(${(confidence*100).toFixed(0)}%)`
            console.log('text ' + text)
            if(confidence > hiconf_threshold)
                  text       = `<b>${text}</b>`
            texts = texts.concat(text)
        }
        const full_text = texts.join(', ') || '-'
        return full_text
    }


    static update_flags(filename){
      const results    = GLOBAL.files[filename]?.results;
      if(!results)
        return
      const flags      = results.compute_flags(filename)

      let   $flag_icon = $(`.table-row[filename="${filename}"]`).find('.lowconf-flag');
      $flag_icon.css('visibility', flags.includes('unsure')? 'visible' : 'hidden')  //hide()/show() changes layout

      const empty      = flags.includes('empty');
      const multiple   = flags.includes('multiple');
            $flag_icon = $(`.table-row[filename="${filename}"]`).find('.amounts-flag');
      $flag_icon.css('visibility', (empty||multiple)? 'visible' : 'hidden')
      if(empty){
        $flag_icon.addClass('outline');         //empty
        $flag_icon.removeClass('checkered');
        $flag_icon.attr('title', 'No detections');
      } else if(multiple) {
        $flag_icon.addClass('checkered');      //multiple
        $flag_icon.removeClass('outline');
        $flag_icon.attr('title', 'Multiple detections');
      }
    }

    static update_all_flags(){
        const filenames = Object.keys(GLOBAL.files)
        for(const filename of filenames) {
            this.update_flags(filename)
        }
    }

    static on_flag(event){
      event.stopPropagation();
      const $cell   = $(event.target).closest('td')
      //const flagged = $cell.attr('manual-flag')
      //console.warn($cell[0], flagged)
      //$cell.attr('manual-flag', !flagged)
      $cell.toggleClass('manual-flag')

      const filename = $cell.closest('[filename]').attr('filename')
      this.update_flags(filename)
    }
}


