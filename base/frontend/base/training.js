
BaseTraining = class BaseTraining{
    static refresh_tab(){
        var $table          = $('#training-filetable')
        $table.find('tbody').html('');

        //refactor
        if($('tbody#training-selected-files').length>0){    
            var processed_files = Object.keys(GLOBAL.files).filter( k => (GLOBAL.files[k].results!=undefined) )
            for(var f of processed_files)
                $('#training-filetable-row').tmpl({filename:f}).appendTo($table.find('tbody#training-selected-files'))
            $table.find('.checkbox').checkbox({onChange: _ => this.update_table_header()})
        }
        
        this.update_table_header()
        this.update_model_info()
    }

    static async on_start_training(){
        var filenames = this.get_selected_files()
        console.log('Training on ', filenames)
        
        const progress_cb = (m => this.on_training_progress(m))
        try {
            this.show_modal()
            await this.upload_training_data(filenames)

            $(GLOBAL.event_source).on('training', progress_cb)
            //FIXME: success/fail should not be determined by this request
            await $.post('/training', JSON.stringify({filenames:filenames, options:this.get_training_options()}))
            if(!$('#training-modal .ui.progress').progress('is complete'))
                this.interrupted_modal()
            
            GLOBAL.App.Settings.load_settings()
        } catch (e) {
            console.error(e)
            this.fail_modal()
        } finally {
            $(GLOBAL.event_source).off('training', progress_cb)
        }
    }

    static get_training_options(){
        return undefined;
    }

    static on_cancel_training(){
        $.get('/stop_training')
            .fail(   _ => $('body').toast({message:'Stopping failed.', class:'error'}) )
            //.always( _ => this.hide_modal() )
        $('#training-modal #cancel-training-button').addClass('disabled')
        return false; //prevent automatic closing of the modal
    }

    static get_selected_files(){
        var $table          = $('#training-filetable')
        var is_selected     = $table.find('.checkbox').map( (i,c) => $(c).checkbox('is checked')).get()
        var filenames       = $table.find('[filename]').map( (i,x) => x.getAttribute('filename') ).get()
        filenames           = filenames.filter( (f,i) => is_selected[i] )
        return filenames
    }

    static update_table_header(){
        $('#training-filetable').find('thead th').text(`Selected ${this.get_selected_files().length} files for training`)
    }

    static update_model_info(model_type='detection'){
        let model_name    = GLOBAL.settings.active_models[model_type]
        const unsaved     = (model_name=='')
        if(unsaved)
            model_name    = '[UNSAVED MODEL]';
        $('#training-new-modelname-field').toggle(unsaved)        //TODO: should be shown also when interrupted
        $('#training-model-info-label').text(model_name)
        $('#training-model-info-message').removeClass('hidden')
    }

    static show_modal(){
        $('#training-modal .progress')
            .progress('remove error')
            .progress('remove success')
            .progress('set label', 'Training in progress...')
            .progress('reset');
        $('#training-modal #ok-training-button').hide()
        $('#training-modal #cancel-training-button').removeClass('disabled').show()
        
        $('#training-modal').modal({
            closable: false, inverted:true, onDeny: x => this.on_cancel_training(),
        }).modal('show');
    }

    static hide_modal(){
        $('#training-modal').modal('hide');
    }

    static fail_modal(){
        $('#training-modal .progress').progress('set error', 'Training failed');
        $('#training-modal #cancel-training-button').removeClass('disabled')
        $('#training-modal').modal({closable:true})
    }

    static interrupted_modal(){
        $('#training-modal .progress').progress('set error', 'Training interrupted');
        $('#training-modal #cancel-training-button').removeClass('disabled')
        $('#training-modal').modal({closable:true})
    }

    static success_modal(){
        $('#training-modal .progress').progress('set success', 'Training finished');
        $('#training-modal #ok-training-button').show()
        $('#training-modal #cancel-training-button').hide()
    }

    static on_training_progress(message){
        var data = JSON.parse(message.originalEvent.data)
        $('#training-modal .progress').progress({percent:data.progress*100, autoSuccess:false})
        $('#training-modal .label').text(data.description)
        if(data.progress >= 1){
            this.success_modal()
            //this.update_model_info()
        }
    }

    static on_save_model(){
        const new_modelname = $('#training-new-modelname')[0].value
        console.log('Saving new model as:', new_modelname)
        $.get('/save_model', {newname: new_modelname, options:this.get_training_options()})
            .done( _ => $('#training-new-modelname-field').hide() )
            .fail( _ => $('body').toast({message:'Saving failed.', class:'error', displayTime: 0, closeIcon: true}) )
        $('#training-new-modelname')[0].value = ''
    }

    static upload_training_data(filenames){
        //TODO: show progress
        var promises      = filenames.map( f => upload_file_to_flask(GLOBAL.files[f]) )
        //TODO: refactor
        //TODO: standardize file name
        var segmentations = filenames.map(    f => GLOBAL.files[f].results.segmentation )
                                     .filter( s => s instanceof Blob )
        promises          = promises.concat( segmentations.map( f => upload_file_to_flask(f) ) )
        return Promise.all(promises).catch( this.fail_modal )  //FIXME: dont catch, handle in calling function
    }
}



ObjectDetectionTraining = class extends BaseTraining {
    //override
    static refresh_tab(){
        super.refresh_tab()
        
        const train_det = $('#train-detector-checkbox').checkbox('is checked')

        $('table#detector-classes').toggle(train_det)

        const callback = (_ => this.refresh_class_selection())
        $('#classes-of-interest-dropdown').dropdown({onChange: callback})
        $('#other-classes-dropdown').dropdown(      {onChange: callback})
        $('#unknown-classes-dropdown').dropdown(    {onChange: callback})
        this.refresh_class_selection()
    }

    //override
    static upload_training_data(filenames){
        //TODO: show progress
        const files           = filenames.map( k => GLOBAL.files[k] )
        const targetfiles     = files.map(
            f => GLOBAL.App.Download.build_annotation_jsonfile(f.name, f.results)
        )

        const promises = files.concat(targetfiles).map( f => upload_file_to_flask(f) )
        return Promise.all(promises).catch( this.fail_modal )
    }

    //override
    static get_training_options(){
        const [coi_selected, oth_selected, unk_selected, rejected] = this.get_class_selection()

        return {
            classes_of_interest : coi_selected,
            classes_other       : oth_selected,
            classes_unknown     : unk_selected,
            classes_rejected    : rejected,
            train_detector      : $('#train-detector-checkbox').checkbox('is checked'),
            learning_rate       : Number($('#training-learning-rate')[0].value),
            epochs              : Number($('#training-number-of-epochs')[0].value),
        };
    }


    static collect_class_counts(){
        const filenames = this.get_selected_files()
        const labels    = filenames
                          .map( f => GLOBAL.files[f].results?.labels )
                          .filter(Boolean)
                          .flat()
                          .map( l => l.trim() || GLOBAL.App.NEGATIVE_CLASS )
        let   label_set = new Set(labels)
        const known_classes = GLOBAL.App.Settings.get_properties_of_active_model()?.['known_classes'] ?? []
        known_classes.map(c => label_set.add(c))
        const uniques   = [...(label_set)].sort()
        const counts    = uniques.map( l => labels.filter( x => x==l ).length )
        return [uniques, counts]
    }

    static get_class_selection(){
        const all_classes  = this.collect_class_counts()[0]
        const coi_selected = $('#classes-of-interest-dropdown').dropdown('get value').split(',')
        const oth_selected = $('#other-classes-dropdown').dropdown('get value').split(',')
        const unk_selected = $('#unknown-classes-dropdown').dropdown('get value').split(',')
        const any_selected = [...coi_selected, ...oth_selected, ...unk_selected]
        const rejected     = all_classes.filter( s => !any_selected.includes(s) )

        return [coi_selected, oth_selected, unk_selected, rejected]
    }

    static refresh_class_count_table(){
        const [classes, counts]  = this.collect_class_counts()

        const [coi_selected, oth_selected, unk_selected, _rejected] = this.get_class_selection()
        const coi_ixs      = Object.keys(classes).filter( i => coi_selected.includes(classes[i]) )
        const oth_ixs      = Object.keys(classes).filter( i => oth_selected.includes(classes[i]) )
        const unk_ixs      = Object.keys(classes).filter( i => unk_selected.includes(classes[i]) )

        const $table       = $('table#detector-classes tbody')
        $table.html('')
        for(const i of coi_ixs){
            //console.log(classes[i], counts[i])
            $(`<tr>
                <td>${classes[i]}</td>
                <td>${counts[i] }</td>
            </tr>`).appendTo($table)
        }
        const coi_count    = coi_ixs.map( i => counts[i] ).reduce( (a,b) => a+b, 0 )
        const oth_count    = oth_ixs.map( i => counts[i] ).reduce( (a,b) => a+b, 0 )
        const unk_count    = unk_ixs.map( i => counts[i] ).reduce( (a,b) => a+b, 0 )
        const rej_count    = counts.reduce( (a,b) => a+b, 0 ) - coi_count - oth_count - unk_count;
        $('#classes-of-interest-count').text(coi_count)
        $('#other-classes-count').text(oth_count)
        $('#unknown-classes-count').text(unk_count)
        $('#rejected-classes-count').text(rej_count)

        $('#detector-positive-classes-count').text(coi_count+oth_count+unk_count)
        $('#detector-negative-classes-count').text(rej_count)
    }

    static refresh_class_selection() {
        const [coi_selected, oth_selected, unk_selected, rejected] = this.get_class_selection()
        
        const $coi_list     = $('#classes-of-interest-list')
        $coi_list.html('')
        rejected.map( s => $(`<div class="item" data-value="${s}">${s}</div>`).appendTo($coi_list) )

        const $oth_list     = $('#other-classes-list')
        $oth_list.html('')
        rejected.map( s => $(`<div class="item" data-value="${s}">${s}</div>`).appendTo($oth_list) )

        const $unk_list     = $('#unknown-classes-list')
        $unk_list.html('')
        rejected.map( s => $(`<div class="item" data-value="${s}">${s}</div>`).appendTo($unk_list) )


        const $rejectedlist = $('#rejected-classes-list')
        $rejectedlist.html('')
        rejected.map( s => $(`<div class="ui label">${s}</div>`).appendTo($rejectedlist) )

        this.refresh_class_count_table()
    }

    static reset_class_selection(){
        const all_classes     = this.collect_class_counts()[0]
        let   known_classes   = GLOBAL.App.Settings.get_properties_of_active_model()?.['known_classes'] ?? []
              known_classes   = known_classes.filter( x => x != GLOBAL.App.NEGATIVE_CLASS )
        let   unknown_classes = all_classes.filter( x => x.includes('unknown') )
                                           .filter( x => x.includes('undeterminable') )

        $('#classes-of-interest-dropdown')
            .dropdown('refresh')
            .dropdown('set selected', known_classes)
        $('#unknown-classes-dropdown')
            .dropdown('refresh')
            .dropdown('set selected', unknown_classes)
    }
}


window.addEventListener(BaseSettings.SETTINGS_CHANGED, () => {
    GLOBAL.App.Training.refresh_tab();
    GLOBAL.App.Training.reset_class_selection?.();
})
