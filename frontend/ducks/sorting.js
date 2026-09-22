

DuckSorting = class extends BaseSorting {
    static on_table_header(event){
        const column_index = super.on_table_header(event)
        if(column_index==1)
            this.on_sort_by_number(event)
        if(column_index==2)
            this.on_sort_by_confidence(event)
    }

    //called when user clicks on "Flags" column head
    static on_sort_by_confidence(event){
        const $col      = $(event.target);
        const direction = $col.hasClass('ascending')? 'descending' : 'ascending';
        this._clear_sorted()
        $col.addClass(['sorted', direction]);

        let   filenames   = Object.keys(GLOBAL.files)
        //unprocessed files have no results and sort after every processed file
        const worstconf   = filenames.map(f => this.lowest_confidence(GLOBAL.files[f].results))

        //sort by the lowest confidence
        const order       = arange(worstconf.length).sort( (a,b) => (worstconf[b] - worstconf[a]) )
        filenames         = order.map(i => filenames[i]);
        if(direction == 'ascending')
            filenames = filenames.reverse()

        this.set_new_file_order(filenames)
    }

    //lowest confidence among a file's predictions; 100 when there are none, -1 when unprocessed
    static lowest_confidence(results){
        if(!results)
            return -1
        const confidences = results.predictions.map( p => Math.max(...Object.values(p)) )
        return confidences.reduce( (carry, x) => Math.min(x, carry), 100 )
    }

    //called when user clicks on "Detected Ducks" column head
    static on_sort_by_number(event){
        const $col       = $(event.target);
        const direction  = $col.hasClass('ascending')? 'descending' : 'ascending';
        this._clear_sorted()
        $col.addClass(['sorted', direction]);

        let   filenames   = Object.keys(GLOBAL.files)
        //unprocessed files count as -1 so they sort after files with zero detections
        const counts      = filenames.map(f => GLOBAL.files[f].results?.labels.length ?? -1)
        const order       = arange(counts.length).sort( (a,b) => (counts[b] - counts[a]) )
        filenames         = order.map(i => filenames[i]);
        if(direction=='ascending')
            filenames = filenames.reverse()

        this.set_new_file_order(filenames)
    }
}
