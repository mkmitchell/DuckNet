
BaseSorting = class {
    static on_table_header(event){
        const columns      = $(event.target).closest('thead').find('th').get()
        const column_index = columns.indexOf(event.target)

        if(column_index==0)
            this.on_sort_by_filename(event)
        return column_index;
    }

    static on_sort_by_filename(event){
        const $col      = $(event.target);
        const direction = $col.hasClass('ascending')? 'descending' : 'ascending';
        this._clear_sorted()
        $col.addClass(['sorted', direction]);
    
        let   filenames = Object.keys(GLOBAL.files).sort()
        if(direction=='descending')
            filenames = filenames.reverse()
        
        this.set_new_file_order(filenames)
    }

    static set_new_file_order(filenames){
        const rows = filenames.map( f => $(`#filetable tr[filename="${f}"]`) );
        $('#filetable tbody').append(rows);
        GLOBAL.files = sort_object(GLOBAL.files, filenames);
    }
    
    static _clear_sorted(){
        $('#filetable .sorted').removeClass(['sorted', 'ascending', 'descending']);
    }
}
