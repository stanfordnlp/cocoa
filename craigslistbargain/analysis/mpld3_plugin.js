!function(mpld3){
       
    mpld3.register_plugin("htmltooltip", HtmlTooltipPlugin);
    HtmlTooltipPlugin.prototype = Object.create(mpld3.Plugin.prototype);
    HtmlTooltipPlugin.prototype.constructor = HtmlTooltipPlugin;
    HtmlTooltipPlugin.prototype.requiredProps = ["id"];
    HtmlTooltipPlugin.prototype.defaultProps = {labels:null, hoffset:0, voffset:10};
    function HtmlTooltipPlugin(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    HtmlTooltipPlugin.prototype.draw = function(){
       var obj = mpld3.get_element(this.props.id);
       var labels = this.props.labels;
       var tooltip = d3.select("body").append("div")
                    .attr("class", "mpld3-tooltip")
                    .style("position", "absolute")
                    .style("z-index", "10")
                    .style("visibility", "hidden");

       obj.elements()
           .on("mouseover", function(d, i){
                              tooltip.html(labels[i])
                                     .style("visibility", "visible");})
           .on("mousemove", function(d, i){
                    tooltip
                      .style("top", d3.event.pageY + this.props.voffset + "px")
                      .style("left",d3.event.pageX + this.props.hoffset + "px");
                 }.bind(this))
           .on("mouseout",  function(d, i){
                           tooltip.style("visibility", "hidden");});
    };
}(mpld3);
