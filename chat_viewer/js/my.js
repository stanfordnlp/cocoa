$.fn.dataTable.ext.search.push(
    function( settings, data, dataIndex ) {
        var check_range = function (min, max, val) {
            if ( ( isNaN( min ) && isNaN( max ) ) ||
                 ( isNaN( min ) && val <= max ) ||
                 ( min <= val   && isNaN( max ) ) ||
                 ( min <= val   && val <= max ) )
            {
                return true;
            }
            return false;
        }

        var check_eq = function (data_val, input_val) {
            if ((!input_val) || (data_val == input_val)) {
                return true;
            }
            return false;
        }

        // num items
        var min = parseInt( $('#items .min').val(), 10 );
        var max = parseInt( $('#items .max').val(), 10 );
        var num_items = parseInt( data[2] ) || 0;
        result = check_range(min, max, num_items);

        // num attrs
        var min = parseInt( $('#attrs .min').val(), 10 );
        var max = parseInt( $('#attrs .max').val(), 10 );
        var num_attrs = parseInt( data[3] ) || 0;
        result = check_range(min, max, num_attrs) && result;

        // agents
        var agent0 = $('#agent0').val()
        var agent1 = $('#agent1').val()
        result = check_eq(data[4], agent0) && result;
        result = check_eq(data[5], agent1) && result;

        var outcome = $('#outcome').val()
        result = check_eq(data[6], outcome) && result;
        
        return result;
    }
);

$(document).ready(function() {
    $(function(){
        var reset = function () {
            $('.min').val('');
            $('.max').val('');
            $('#agent0').val('');
            $('#agent1').val('');
            $('#inputSenarioId').val('');
            $('#outcome').val('');
        }
        $.getJSON('../metadata.json', function (data) {
            reset();
            var columns = [
                        {"data": "dialogue_id"},
                        {"data": "scenario_id"},
                        {"data": "num_items"},
                        {"data": "num_attrs"},
                        {"data": "agent0"},
                        {"data": "agent1"},
                        {"data": "outcome"}
                        ];
            var columnDefs = [
                        {"targets": 0, "title": "Dialogue",
                         "render": function ( data, type, full, meta ) {
                             return data.substr(0, 8);
                         }
                        },
                        {"targets": 1, "title": "Scenario ID",
                         "render": function ( data, type, full, meta ) {
                             return data.substr(0, 8);
                         }
                        },
                        {"targets": 2, "title": "#Items"},
                        {"targets": 3, "title": "#Attrs"},
                        {"targets": 4, "title": "Agent 0"},
                        {"targets": 5, "title": "Agent 1"},
                        {"targets": 6, "title": "Outcome"},
                        ];
            if ('fluent' in data['data'][0]) {
                columns.push({"data": "fluent"});
                columns.push({"data": "correct"});
                columns.push({"data": "cooperative"});
                columns.push({"data": "humanlike"});
                columnDefs.push({"targets": 7, "title": "Fluent", "render": $.fn.dataTable.render.number( ',', '.', 1)});
                columnDefs.push({"targets": 8, "title": "Correct", "render": $.fn.dataTable.render.number(',', '.', 1)});
                columnDefs.push({"targets": 9, "title": "Cooperative", "render": $.fn.dataTable.render.number(',', '.', 1)});
                columnDefs.push({"targets": 10, "title": "Human-like", "render": $.fn.dataTable.render.number(',', '.', 1)});
            }

            var table = $('#metadata').DataTable( {
                    "ajax": '../metadata.json',
                    "dom": '<"top"l>rt<"bottom"ip><"clear">',
                    "columns": columns,
                    "columnDefs": columnDefs
            } );

            $('#metadata tbody').on('click', 'tr', function() {
                var d = table.row( this ).data();
                $('#example').load('../chat_htmls/' + d.dialogue_id + '.html');

                if ( $(this).hasClass('selected') ) {
                    $(this).removeClass('selected');
                }
                else {
                    table.$('tr.selected').removeClass('selected');
                    $(this).addClass('selected');
                }

            });
    
            $('#filterButton').click(function() {
                //alert("button");
                var searchText = $('#inputScenarioId').val()
                if (searchText) {
                    table.search(searchText).draw() ;
                }
                table.draw();
            } );

            $('#resetButton').click(function() {
                reset();
                table.draw();
            } );
        } );
    });
} );
