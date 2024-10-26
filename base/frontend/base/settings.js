

class BaseSettings {
    static SETTINGS_CHANGED = 'settings-changed'

    static async load_settings() {
        //TODO: error handling
        const data = await $.get('/settings')
        
        const settings = data.settings
        const models   = data.available_models
        console.log('Loaded settings:  ', settings)
        console.log('Available models: ', models)
        GLOBAL.settings         = settings
        GLOBAL.available_models = data.available_models
        BaseSettings.dispatch_event()
        this.update_settings_modal(models)
        
        return data;
    }

    static update_settings_modal(models) {
        const settings = GLOBAL.settings;

        if (models.detection)
            this.update_model_selection_dropdown(
                models.detection, settings.active_models.detection, $("#settings-active-model")
            )

        // // Apply other settings to the UI
        // console.log("Confidence Threshold:", settings.confidence_threshold); // Add this line
        // console.log("Flag Negatives:", settings.flag_negatives); // Add this line
        // console.log("Export Boxes:", settings.export_boxes); // Add this line

        // $("#settings-confidence-threshold-input")[0].value = settings.confidence_threshold;
        // $("#settings-flag-negatives").checkbox(settings.flag_negatives ? 'check' : 'uncheck');
        // $("#settings-export-boxes").checkbox(settings.export_boxes ? 'check' : 'uncheck');
    }

    static update_model_selection_dropdown(models, active_model, $dropdown) {
        const modelnames   = models.map(x => ((typeof x) == 'string') ? x : x.name)
        let dropdown_items = modelnames.map((x, i) => {
            return { name: x, value: x, selected: (x == active_model) };
        })
        if (active_model == '')
            dropdown_items.push({ name: '[UNSAVED MODEL]', value: '', selected: true })

        $dropdown.dropdown({
            values: dropdown_items,
            showOnFocus: false,
            onChange: (dropdown_value) => {
                const i = modelnames.indexOf(dropdown_value)
                const properties = models[i]?.properties;
                this.display_model_properties(properties, $dropdown)
            }
        })

        const i = modelnames.indexOf($dropdown.dropdown('get value'))
        this.display_model_properties(models[i]?.properties, $dropdown)
    }

    static apply_settings_from_modal() {
        const value = $("#settings-active-model").dropdown('get value');
        // console.log("Applying settings with value:", value); // Add this line
        // GLOBAL.settings.active_models.detection = value;
        // GLOBAL.settings.confidence_threshold = Number($("#settings-confidence-threshold-input")[0].value);
        // GLOBAL.settings.flag_negatives = $("#settings-flag-negatives").checkbox('is checked');
        // GLOBAL.settings.export_boxes = $("#settings-export-boxes").checkbox('is checked');
    }


    static on_save_settings(_){
        this.apply_settings_from_modal()
        this.dispatch_event()

        var settingsdata = deepcopy(GLOBAL.settings);
        var postdata     = JSON.stringify(settingsdata);
        $('#settings-ok-button').addClass('loading');

        $.post(`/settings`, postdata,).done( x => {
            $('#settings-dialog').modal('hide');
            console.log('Settings saved successfully:',x)
        }).fail( x => {
            console.error('Saving settings failed', x)
            $('body').toast({message:'Saving failed', class:'error'})
        }).always( _ => {
            $('#settings-ok-button').removeClass('loading');
        } );

        //do not close the dialog, doing this manually in the .post() callback
        return false;
    }

    static on_settings(){
        this.load_settings();
        $('#settings-dialog').modal({onApprove: _ => this.on_save_settings()}).modal('show');
    }

    static dispatch_event(){
        const event = new CustomEvent('settings-changed', GLOBAL.settings);
        window.dispatchEvent(event)
    }

    static display_model_properties(properties, $dropdown){
        const $known_classes = $dropdown.siblings('.known-classes').first()
        if(!properties){
            $known_classes.hide()
        } else {
            $known_classes.find('.label').remove()
            $known_classes.show().append(
                properties['known_classes']
                    .filter( x => x.toLowerCase() != GLOBAL.App.NEGATIVE_CLASS.toLowerCase() )
                    .map(c => `<div class="ui label">${c}</div>`)
            )
        }
    }

    static get_properties_of_active_model(modeltype='detection'){
        const modelname = GLOBAL.settings['active_models'][modeltype]
        const model     = GLOBAL.available_models[modeltype].filter(x => x.name==modelname)[0]
        return model?.properties;
    }
}