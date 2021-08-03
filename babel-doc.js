let functions={};
export {functions};
export default function module(babel){    
    const _babelName = 'bable-plugin-doc';

    function buildReadMe(funName, fun,chunks) {    
        let funnode=fun.node;  
        let str='';
        str+=funName;
        //str+=funName+":"+fun.getSource().substring(0,fun.node.body.start-fun.node.start);
        /*str+=" function ";
        if(funnode.generator){
            str+='*';
        }    */    
        str+='(';
        for(let i=0;i<funnode.params.length;++i){
            let v=funnode.params[i];
            if(i!=0){
                str+=",";
            }            
            if(v.type=="RestElement"){
                str+='...';
                v=v.argument;
            }
            if(v.type=="AssignmentPattern"){
                str+=v.left.name+'=';
                v=v.right;
            }
            if(v.type=="NumericLiteral"){
                str+=v.value;
            }
            if(v.type=="Identifier"){
                str+=v.name;
            }            
        }
        str+=')';
        return str+"\n"+chunks.trim().replace(/(\s?\*\s)*/g,'').trim();
    }

    let t=babel.types;
    function walk(tempName, fun, curNode) {
        if (curNode.leadingComments && curNode.leadingComments.length && curNode.leadingComments.some(o => o.type === 'CommentBlock')) {
            if(tempName!='')functions[tempName]=( buildReadMe(tempName,fun, curNode.leadingComments[curNode.leadingComments.length-1].value) );
        }else{
            if(tempName!='')functions[tempName]=( buildReadMe(tempName,fun, '') );
        }
    }

    const visitor = {
        name: _babelName,
        pre() {
           //console.log('');
        },
        visitor: {
            FunctionDeclaration(path) {                
                walk(path.node.id.name,path,path.node);
            },
            AssignmentExpression(path) {
                let node=path.node;
                let right=node.right;
                let left=node.left;
                if(right.type=='FunctionExpression'){
                    walk(path.get('left').getSource(),path.get('right'),path.getStatementParent().node);
                }
            },
        },
        post(state) {  
            //console.log('');
        }
    }
    return visitor;
};